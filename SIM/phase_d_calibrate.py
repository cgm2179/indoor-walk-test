#!/usr/bin/env python3
"""
Phase D — Calibration against G-flex scanner measurements (spec §6).

STATUS: scaffold, blocked on the right measurement data. The existing walk
CSV recorded OUTDOOR macro donors at unknown sites; the calibration loop
needs either (a) a re-walk with a known transmitter at a known indoor
position and known P_tx, or (b) donor site coordinates for the BS mode.
When that data exists, fill MEASUREMENTS_CSV below and run.

Expected CSV columns: x_px, y_px (model-grid cells, or lon/lat -> convert
with the STEP_1 affines), rsrp_dbm, freq_mhz. Multiple rows per cell are
median-combined (D.1 fast-fading suppression).

usage:
  python SIM/phase_d_calibrate.py --fit          # least-squares parameter fit
  python SIM/phase_d_calibrate.py --sanity      # D.3 checks on the simulator
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pa", SIM / "phase_a.py")
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)

MEASUREMENTS_CSV = SIM / "measurements_known_tx.csv"   # <- provide this
TX_KNOWN = dict(x=246.0, y=189.0, p_tx_dbm=None)       # <- and this


def fit():
    import pandas as pd
    from scipy.optimize import least_squares
    if not MEASUREMENTS_CSV.exists() or TX_KNOWN["p_tx_dbm"] is None:
        raise SystemExit(
            "BLOCKED: needs a measurement walk with a known transmitter.\n"
            f"Provide {MEASUREMENTS_CSV.name} (x_px, y_px, rsrp_dbm, freq_mhz)"
            " and set TX_KNOWN. See SIM/README.md 'Phase D'.")

    grid = np.load(SIM / "grid_model.npy")
    manifest = json.loads((SIM / "manifest.json").read_text())
    cell = manifest["cell_size_m"]
    df = pd.read_csv(MEASUREMENTS_CSV)
    df["cx"], df["cy"] = df.x_px.round().astype(int), df.y_px.round().astype(int)
    df = df.groupby(["cx", "cy", "freq_mhz"], as_index=False).rsrp_dbm.median()
    pl_meas = TX_KNOWN["p_tx_dbm"] - df.rsrp_dbm.values

    # 20% honest holdout (D.2.4)
    rng = np.random.default_rng(0)
    hold = rng.random(len(df)) < 0.2

    base = np.array([m["loss_db"] for m in pa.MATERIALS6])

    def residuals(theta):
        n_exp, scale = theta[0], theta[1:]
        pa.N_EXP = n_exp
        jitter = np.ones(6)
        jitter[1:] = scale                       # per-material scale factor
        res = []
        for f in sorted(df.freq_mhz.unique()):
            m = (~hold) & (df.freq_mhz.values == f)
            if not m.any():
                continue
            pl = pa.pathloss_map(grid, (TX_KNOWN["x"], TX_KNOWN["y"]),
                                 float(f), cell, jitter=jitter)
            res.append(pl_meas[m] - pl[df.cy.values[m], df.cx.values[m]])
        return np.concatenate(res)

    x0 = np.concatenate([[2.0], np.ones(5)])
    lb = np.concatenate([[1.5], np.full(5, 0.5)])   # bounds: +/-50% (D.2.2)
    ub = np.concatenate([[3.0], np.full(5, 1.5)])
    sol = least_squares(residuals, x0, bounds=(lb, ub), verbose=1)
    n_fit, scales = sol.x[0], sol.x[1:]
    print(f"n = {n_fit:.2f}")
    for m, s in zip(pa.MATERIALS6[1:], scales):
        print(f"  {m['name']:<20} {m['loss_db']:5.1f} -> {m['loss_db']*s:5.1f} dB")
    r = residuals(sol.x)
    print(f"fit residual RMSE {np.sqrt((r**2).mean()):.2f} dB")
    # honest holdout error at 3.5 GHz nearest map
    out = dict(n_exp=float(n_fit),
               material_scale={m["name"]: float(s)
                               for m, s in zip(pa.MATERIALS6[1:], scales)},
               fit_rmse_db=float(np.sqrt((r**2).mean())))
    (SIM / "calibration.json").write_text(json.dumps(out, indent=2))
    print("wrote calibration.json -> rerun Phase B with these values, then "
          "fine-tune the UNet 20 epochs (spec D.2.3)")


def sanity():
    """D.3 checks, runnable today against the simulator itself."""
    grid = np.load(SIM / "grid_model.npy")
    inside = np.load(SIM / "inside_mask.npy")
    manifest = json.loads((SIM / "manifest.json").read_text())
    cell = manifest["cell_size_m"]
    tx = (246.0, 189.0)
    pl, k = pa.pathloss_map(grid, tx, 3500.0, cell, return_crossings=True)

    los = (k == 0) & inside
    ray = pl[int(tx[1]), int(tx[0]):int(tx[0]) + 60]
    ok1 = bool((np.diff(ray) >= -1e-3).mean() > 0.95)
    print(f"[{'PASS' if ok1 else 'FAIL'}] monotone decay along LOS ray")

    d = np.maximum(np.hypot(*np.meshgrid(
        np.arange(grid.shape[1]) - tx[0],
        np.arange(grid.shape[0]) - tx[1])[::1]) * cell, 1.0)
    fspl = pa.fspl_1m_db(3500) + 20 * np.log10(d)
    below = (pl < fspl - 0.1)[inside].mean()
    print(f"[{'PASS' if below < 0.001 else 'FAIL'}] no cell below FSPL "
          f"({below * 100:.3f}%)")

    pl24 = pa.pathloss_map(grid, tx, 2442.0, cell)
    frac = ((pl24 <= pl + 1e-3)[inside]).mean()
    print(f"[{'PASS' if frac > 0.99 else 'FAIL'}] 2.4 GHz <= 3.5 GHz cell-wise "
          f"({frac * 100:.1f}%)")

    L, _ = pa.loss_table(3500.0)
    core_step = L[3]
    ys, xs = np.nonzero((grid == 3))
    print(f"[INFO] core loss per crossing at 3.5 GHz: {core_step:.1f} dB "
          f"(shadow-step check needs a specific core; see preview maps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    a = ap.parse_args()
    if a.sanity:
        sanity()
    if a.fit:
        fit()
