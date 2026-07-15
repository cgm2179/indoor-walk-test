#!/usr/bin/env python3
"""
Multi-band signal-quality maps + simulated walk across the floor.

For each band/protocol (matching what the scanner saw: B71/B2 LTE, n41/n77
NR), computes four metric maps from the two candidate transmitters:

  RSRP  best-server reference-signal power per RE   (dBm)
  RSSI  wideband power: both cells + noise, full load (dBm)
  RSRQ  N_RB * RSRP / RSSI                          (dB, ceiling ~-10.8)
  SINR  serving RE power over other-cell + noise    (dB)

The two Tx are modeled co-channel at full load, so the non-serving cell is
interference -- that is what shapes the SINR map, not coverage.

Physics per band: FSPL scales 20 log10(f); material losses get a simple
linear-in-GHz slope about the 3.5 GHz reference values (drywall nearly flat,
concrete/core strongly rising -- rough but directionally right); noise per RE
follows the subcarrier spacing (15 kHz LTE, 30 kHz NR).

Also simulates a walk along the corridor loop and writes a per-band metric
trace (CSV + PNG) and an animated GIF of the receiver moving on the floor.

usage: python STEP_2/multiband_metrics.py [--eirp-dbm 23] [--gif-band NR_n77_3750MHz]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("mk", HERE / "motley_keenan.py")
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)

BANDS = [
    dict(label="LTE_B71_617MHz",  f_mhz=617.0,  protocol="LTE", bw_mhz=10, scs_khz=15),
    dict(label="LTE_B2_1960MHz",  f_mhz=1960.0, protocol="LTE", bw_mhz=20, scs_khz=15),
    dict(label="NR_n41_2506MHz",  f_mhz=2506.0, protocol="NR",  bw_mhz=60, scs_khz=30),
    dict(label="NR_n77_3750MHz",  f_mhz=3750.0, protocol="NR",  bw_mhz=100, scs_khz=30),
]

# dB-per-GHz slope of material loss about the 3.5 GHz reference (rough,
# ITU-P.2040-flavored: gypsum/glass nearly flat, concrete strongly rising)
LOSS_SLOPE_PER_GHZ = {1: 0.3, 2: 2.0, 3: 2.5, 5: 0.3, 6: 0.3, 7: 0.0}
BULK_SLOPE_PER_GHZ = {4: 0.05}
NOISE_FIGURE_DB = 7.0

WALK_WAYPOINTS_PX = [(90, 200), (1040, 200), (1040, 320), (90, 320), (90, 200)]
WALK_SPEED_MPS = 1.4


def band_loss_vectors(mats, f_mhz):
    loss_db = np.zeros(256, np.float32)
    loss_per_m = np.zeros(256, np.float32)
    df = f_mhz / 1000.0 - 3.5
    for k, v in mats.items():
        i = int(k)
        loss_db[i] = max(v["loss_db"] + LOSS_SLOPE_PER_GHZ.get(i, 0) * df,
                         0.3 * v["loss_db"])
        ref = v.get("loss_per_m_db", 0.0)
        loss_per_m[i] = max(ref + BULK_SLOPE_PER_GHZ.get(i, 0) * df, 0.3 * ref)
    return loss_db, loss_per_m


def metrics_for_band(band, grid, mats, txs, mpp, eirp_dbm):
    n_rb = int(band["bw_mhz"] * 1000 * 0.9 / (12 * band["scs_khz"]))
    eirp_re = eirp_dbm - 10 * np.log10(12 * n_rb)          # power per RE
    noise_re = -174 + 10 * np.log10(band["scs_khz"] * 1e3) + NOISE_FIGURE_DB

    loss_db, loss_per_m = band_loss_vectors(mats, band["f_mhz"])
    rsrp_tx = [eirp_re - mk.compute_pathloss(
        grid, loss_db, loss_per_m, (t["x_px"], t["y_px"]), mpp,
        freq_mhz=band["f_mhz"]) for t in txs]

    stack = np.stack(rsrp_tx)                              # (n_tx, H, W)
    serving = stack.argmax(0)
    rsrp = stack.max(0)
    interf_lin = (10 ** (stack / 10)).sum(0) - 10 ** (rsrp / 10)
    sinr = rsrp - 10 * np.log10(interf_lin + 10 ** (noise_re / 10))
    rssi = 10 * np.log10(12 * n_rb) + 10 * np.log10(
        (10 ** (stack / 10)).sum(0) + 10 ** (noise_re / 10))
    rsrq = 10 * np.log10(n_rb) + rsrp - rssi
    return dict(rsrp=rsrp, rsrq=rsrq, sinr=sinr, rssi=rssi,
                serving=serving, n_rb=n_rb)


def render_band(band, m, grid, outside, mpp, out_png):
    H, W = grid.shape
    walls = np.isin(grid, [1, 2, 3, 5, 6, 7])
    panels = [("rsrp", "RSRP (dBm)", -125, -55, "turbo"),
              ("rsrq", "RSRQ (dB)", -20, -10, "viridis"),
              ("sinr", "SINR (dB)", -5, 30, "turbo"),
              ("rssi", "RSSI (dBm)", -95, -25, "turbo")]
    fig, axes = plt.subplots(4, 1, figsize=(11, 13))
    for ax, (key, ttl, lo, hi, cm) in zip(axes, panels):
        cmap = plt.get_cmap(cm).copy()
        cmap.set_bad((0.93, 0.93, 0.93))
        shown = np.where(outside, np.nan, m[key])
        im = ax.imshow(shown, cmap=cmap, vmin=lo, vmax=hi,
                       extent=[0, W * mpp, H * mpp, 0])
        ov = np.zeros((H, W, 4))
        ov[walls] = (0, 0, 0, 0.8)
        ax.imshow(ov, extent=[0, W * mpp, H * mpp, 0])
        ax.set_title(f"{band['label']}  {ttl}", fontsize=10)
        ax.set_xticks([]) if key != "rssi" else ax.set_xlabel("meters")
        fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def walk_route(mpp, dt_s=0.5):
    """Positions along the corridor loop at walking speed."""
    pts = np.array(WALK_WAYPOINTS_PX, float)
    segs = np.diff(pts, axis=0)
    seg_len_m = np.hypot(segs[:, 0], segs[:, 1]) * mpp
    total_s = seg_len_m.sum() / WALK_SPEED_MPS
    ts = np.arange(0, total_s, dt_s)
    cum = np.concatenate([[0], np.cumsum(seg_len_m)])
    d = ts * WALK_SPEED_MPS
    xy = np.empty((len(ts), 2))
    for i, di in enumerate(d):
        s = min(np.searchsorted(cum, di, "right") - 1, len(segs) - 1)
        f = (di - cum[s]) / seg_len_m[s]
        xy[i] = pts[s] + f * segs[s]
    return ts, xy


def animate_walk(band, m, grid, mpp, ts, xy, out_gif):
    from matplotlib.animation import FuncAnimation, PillowWriter
    H, W = grid.shape
    walls = np.isin(grid, [1, 2, 3, 5, 6, 7])
    fig, (ax, axm) = plt.subplots(
        2, 1, figsize=(9.5, 6.4), height_ratios=[2.4, 1])
    ax.imshow(np.where(walls, 0.15, 1.0), cmap="gray", vmin=0, vmax=1,
              extent=[0, W * mpp, H * mpp, 0])
    trail, = ax.plot([], [], "-", color="tab:blue", lw=1.5, alpha=0.6)
    dot, = ax.plot([], [], "o", color="tab:red", ms=9)
    label = ax.text(0.01, 0.97, "", transform=ax.transAxes, fontsize=9,
                    va="top", family="monospace",
                    bbox=dict(fc="white", alpha=0.85, ec="none"))
    ax.set_title(f"simulated walk — {band['label']}", fontsize=11)
    ax.set_xlabel("meters")

    ix = np.clip(xy[:, 0].round().astype(int), 0, W - 1)
    iy = np.clip(xy[:, 1].round().astype(int), 0, H - 1)
    series = {k: m[k][iy, ix] for k in ("rsrp", "rsrq", "sinr", "rssi")}
    axm.plot(ts, series["rsrp"], label="RSRP dBm")
    axm.plot(ts, series["sinr"], label="SINR dB")
    cursor = axm.axvline(0, color="tab:red", lw=1)
    axm.legend(fontsize=8, loc="lower left")
    axm.set_xlabel("walk time (s)")
    axm.grid(alpha=0.3)
    fig.tight_layout()

    step = 2                                   # one frame per second of walk
    frames = range(0, len(ts), step)

    def draw(i):
        trail.set_data(xy[:i + 1, 0] * mpp, xy[:i + 1, 1] * mpp)
        dot.set_data([xy[i, 0] * mpp], [xy[i, 1] * mpp])
        cursor.set_xdata([ts[i]])
        label.set_text(
            f"t={ts[i]:5.1f}s  RSRP {series['rsrp'][i]:7.1f} dBm  "
            f"RSRQ {series['rsrq'][i]:6.1f} dB\n"
            f"SINR {series['sinr'][i]:6.1f} dB   RSSI {series['rssi'][i]:7.1f} dBm")
        return trail, dot, cursor, label

    anim = FuncAnimation(fig, draw, frames=frames, blit=True)
    anim.save(out_gif, writer=PillowWriter(fps=8), dpi=80)
    plt.close(fig)
    return series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eirp-dbm", type=float, default=23.0)
    ap.add_argument("--gif-band", default="NR_n77_3750MHz")
    args = ap.parse_args()

    out = HERE / "multiband"
    out.mkdir(exist_ok=True)
    grid = np.load(HERE / "material_grid_consolidated.npy")
    outside = np.load(HERE / "outside_mask.npy")
    mats = json.loads((ROOT / "STEP_1" / "materials.json").read_text())
    meta = json.loads((ROOT / "STEP_1" / "floorplan_meta.json").read_text())
    txs = json.loads((ROOT / "STEP_1" / "transmitters.json").read_text())["transmitters"]
    mpp = meta["meters_per_px"]
    indoor = ~outside

    ts, xy = walk_route(mpp)
    walk_csv = {}
    for band in BANDS:
        m = metrics_for_band(band, grid, mats, txs, mpp, args.eirp_dbm)
        np.savez_compressed(out / f"metrics_{band['label']}.npz",
                            **{k: v for k, v in m.items() if k != "n_rb"})
        render_band(band, m, grid, outside, mpp, out / f"maps_{band['label']}.png")
        med = {k: float(np.median(m[k][indoor])) for k in
               ("rsrp", "rsrq", "sinr", "rssi")}
        print(f"{band['label']:<18} indoor medians  "
              f"RSRP {med['rsrp']:7.1f}  RSRQ {med['rsrq']:6.1f}  "
              f"SINR {med['sinr']:6.1f}  RSSI {med['rssi']:7.1f}")

        ix = np.clip(xy[:, 0].round().astype(int), 0, grid.shape[1] - 1)
        iy = np.clip(xy[:, 1].round().astype(int), 0, grid.shape[0] - 1)
        for k in ("rsrp", "rsrq", "sinr", "rssi"):
            walk_csv[f"{band['label']}_{k}"] = m[k][iy, ix]
        if band["label"] == args.gif_band:
            animate_walk(band, m, grid, mpp, ts, xy,
                         out / f"walk_sim_{band['label']}.gif")

    # walk trace: CSV + comparison figure across bands
    import csv
    with open(out / "walk_trace.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        cols = ["t_s", "x_px", "y_px"] + list(walk_csv)
        wr.writerow(cols)
        for i in range(len(ts)):
            wr.writerow([round(ts[i], 1), round(xy[i, 0], 1), round(xy[i, 1], 1)]
                        + [round(float(walk_csv[c][i]), 2) for c in list(walk_csv)])

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, key, unit in zip(axes, ("rsrp", "rsrq", "sinr", "rssi"),
                             ("dBm", "dB", "dB", "dBm")):
        for band in BANDS:
            ax.plot(ts, walk_csv[f"{band['label']}_{key}"],
                    label=band["label"], lw=1.2)
        ax.set_ylabel(f"{key.upper()} ({unit})")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=4)
    axes[0].set_title("simulated corridor-loop walk, all bands")
    axes[-1].set_xlabel("walk time (s)")
    fig.tight_layout()
    fig.savefig(out / "walk_trace_all_bands.png", dpi=120)
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
