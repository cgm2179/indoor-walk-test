#!/usr/bin/env python3
"""
Phase E — Transmitter-placement optimizer (spec §7): exhaustive search over
walkable candidate cells, explicit objectives, top-5 returned (never argmax).

Uses the Phase A physics directly (ground truth, ~0.5 s/candidate) or the
ONNX surrogate when SIM/web/pl_unet.onnx exists (~ms/candidate). The browser
Simulator tab implements the same search in JS; this script is the offline /
reproducible version for the proposal figures.

Objectives (7.2):
  coverage      fraction of walkable cells with P_rx - M_sigma >= threshold
  mean_pl       mean path loss over walkable cells (the literal ask)
  hole_filling  coverage restricted to cells where the BS map is weak

usage:
  python SIM/phase_e_optimizer.py --objective coverage --stride 4 \\
      --freq 3500 --tx-power 23 --gain 3 [--reliability 90] [--physics]
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pa", SIM / "phase_a.py")
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)

Z = {"80": 0.84, "90": 1.28, "95": 1.65}


def pl_from_onnx(sess, manifest, onehot, tx, freq_feat):
    H, W = manifest["grid_shape"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    s = manifest["tx_blob_sigma_cells"]
    cell = manifest["cell_size_m"]
    dn = manifest.get("dist_channel_norm", 3.0)
    d = np.hypot(xx - tx[0], yy - tx[1])
    x = np.empty((1, 9, H, W), np.float32)
    x[0, :6] = onehot
    x[0, 6] = np.exp(-(d ** 2) / (2 * s * s))
    x[0, 7] = freq_feat
    x[0, 8] = np.log10(np.maximum(d * cell, 1.0)) / dn
    out = sess.run(None, {"x": x})[0][0]
    n = manifest["norm"]
    return np.clip(out, 0, 1) * n["pl_range_db"] + n["pl_min_db"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="coverage",
                    choices=["coverage", "mean_pl", "hole_filling"])
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--freq", type=float, default=3500.0)
    ap.add_argument("--tx-power", type=float, required=True, help="dBm (R10)")
    ap.add_argument("--gain", type=float, required=True, help="dBi Tx+Rx (R10)")
    ap.add_argument("--threshold", type=float, default=-85.0)
    ap.add_argument("--reliability", default="90", choices=list(Z))
    ap.add_argument("--sigma-sf", type=float, default=8.03,
                    help="shadow-fading sigma (Phase D residuals when known)")
    ap.add_argument("--physics", action="store_true",
                    help="force the Phase A generator even if ONNX exists")
    ap.add_argument("--bs-map", help="npy of BS P_rx (for hole_filling)")
    args = ap.parse_args()

    manifest = json.loads((SIM / "manifest.json").read_text())
    grid = np.load(SIM / "grid_model.npy")
    walk = np.load(SIM / "walkable_mask.npy")
    cell = manifest["cell_size_m"]
    onehot = np.stack([(grid == c) for c in range(6)]).astype(np.float32)

    onnx_path = SIM / "web" / "pl_unet.onnx"
    sess = None
    if onnx_path.exists() and not args.physics:
        import onnxruntime
        sess = onnxruntime.InferenceSession(str(onnx_path))
        print("using ONNX surrogate")
    else:
        print("using Phase A physics (slow but exact); --stride big is wise")

    n = manifest["norm"]
    freq_feat = (np.log10(args.freq) - np.log10(n["freq_log_lo_mhz"])) / \
        (np.log10(n["freq_log_hi_mhz"]) - np.log10(n["freq_log_lo_mhz"]))
    margin = Z[args.reliability] * args.sigma_sf
    eirp = args.tx_power + args.gain

    target_mask = walk.copy()
    if args.objective == "hole_filling":
        assert args.bs_map, "--bs-map required for hole_filling"
        bs = np.load(args.bs_map)
        target_mask &= bs < args.threshold
        print(f"hole cells: {target_mask.sum()}")

    ys, xs = np.nonzero(walk)
    cand = [(x, y) for y, x in zip(ys, xs)
            if x % args.stride == 0 and y % args.stride == 0]
    print(f"{len(cand)} candidates (stride {args.stride})")

    scores = []
    t0 = time.time()
    for i, (x, y) in enumerate(cand):
        if sess is not None:
            pl = pl_from_onnx(sess, manifest, onehot, (x, y), freq_feat)
        else:
            pl = pa.pathloss_map(grid, (float(x), float(y)), args.freq, cell)
        prx = eirp - pl
        if args.objective == "mean_pl":
            score = -pl[walk].mean()                     # higher is better
        else:
            score = (prx[target_mask] - margin >= args.threshold).mean()
        scores.append(score)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(cand)}  ({(time.time()-t0)/(i+1):.2f} s/cand)")

    order = np.argsort(scores)[::-1][:5]                 # top-5, never argmax
    print(f"\nobjective={args.objective}  threshold={args.threshold} dBm  "
          f"margin={margin:.1f} dB  EIRP={eirp} dBm")
    result = []
    for r, i in enumerate(order, 1):
        x, y = cand[i]
        print(f"  #{r}: cell ({x}, {y}) = ({x*cell:.1f}, {y*cell:.1f}) m   "
              f"score {scores[i]:.4f}")
        result.append(dict(rank=r, x_cell=int(x), y_cell=int(y),
                           score=float(scores[i])))
    (SIM / f"optimizer_top5_{args.objective}.json").write_text(json.dumps(dict(
        objective=args.objective, freq_mhz=args.freq, eirp_dbm=eirp,
        threshold_dbm=args.threshold, margin_db=margin,
        stride=args.stride, top5=result), indent=2))


if __name__ == "__main__":
    main()
