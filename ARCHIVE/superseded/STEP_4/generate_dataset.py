#!/usr/bin/env python3
"""
Step 4b: Generate (floor + Tx location -> path-loss map) training pairs for a
RadioUNet-style surrogate network, using the STEP_2 Motley-Keenan engine as
the ground-truth simulator.

Each sample:
  X[0]  per-cell wall loss (dB)         -- the "what's here" channel
  X[1]  per-cell clutter loss (dB/m)
  X[2]  log-distance to Tx (normalized) -- the "where's the Tx" channel
  y     path loss (dB), wall_sat applied

Tx positions are sampled uniformly over indoor air cells, so the network sees
the transmitter everywhere on the floor, not just at the two green pins.
The grid is 2x-downsampled (walls preserved by max-loss pooling) to speed up
generation; at ~2 s/sample, 500 samples is ~15 min of CPU.

usage: python STEP_4/generate_dataset.py [--n-samples 100] [--out STEP_4/dataset]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "mk", ROOT / "STEP_2" / "motley_keenan.py")
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)


def downsample_max_loss(grid, loss_db, f=2):
    """Blockwise pooling that keeps the highest-loss material in each block,
    so 2-px walls survive the downsample instead of vanishing."""
    H, W = (grid.shape[0] // f) * f, (grid.shape[1] // f) * f
    blocks = grid[:H, :W].reshape(H // f, f, W // f, f)
    pri = loss_db[blocks] + 0.001 * (blocks == 4)   # clutter beats plain air
    b2 = blocks.transpose(0, 2, 1, 3).reshape(H // f, W // f, f * f)
    p2 = pri.transpose(0, 2, 1, 3).reshape(H // f, W // f, f * f)
    return np.take_along_axis(b2, p2.argmax(-1)[..., None], -1)[..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "STEP_4" / "dataset"))
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    grid = np.load(ROOT / "STEP_2" / "material_grid_consolidated.npy")
    outside = np.load(ROOT / "STEP_2" / "outside_mask.npy")
    mats = json.loads((ROOT / "STEP_1" / "materials.json").read_text())
    meta = json.loads((ROOT / "STEP_1" / "floorplan_meta.json").read_text())
    sim = json.loads((ROOT / "STEP_2" / "sim_params.json").read_text())

    loss_db = np.zeros(256, np.float32)
    loss_per_m = np.zeros(256, np.float32)
    for k, v in mats.items():
        loss_db[int(k)] = v["loss_db"]
        loss_per_m[int(k)] = v.get("loss_per_m_db", 0.0)

    f = args.downsample
    g = downsample_max_loss(grid, loss_db, f)
    ind = ~outside[: (grid.shape[0] // f) * f: f, : (grid.shape[1] // f) * f: f]
    mpp = meta["meters_per_px"] * f
    H, W = g.shape

    open_cells = np.argwhere((g == 0) & ind)        # (y, x) candidates for Tx
    rng = np.random.default_rng(args.seed)
    picks = open_cells[rng.choice(len(open_cells), args.n_samples, replace=False)]

    wall_map = loss_db[g].astype(np.float32)
    clutter_map = loss_per_m[g].astype(np.float32)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))

    for s, (ty, tx_) in enumerate(picks):
        pl = mk.compute_pathloss(
            g, loss_db, loss_per_m, (float(tx_), float(ty)), mpp,
            freq_mhz=sim["freq_mhz"], n_exp=sim["n_exp"],
            wall_sat_db=sim["wall_sat_db"])
        d = np.hypot(gx - tx_, gy - ty) * mpp
        x2 = np.log10(np.maximum(d, 1.0)).astype(np.float32)
        np.savez_compressed(
            out / f"sample_{s:04d}.npz",
            x=np.stack([wall_map, clutter_map, x2]),
            y=pl.astype(np.float16),
            tx_px=np.array([tx_, ty], np.float32), indoor=ind)
        if (s + 1) % 10 == 0 or s == 0:
            print(f"  {s + 1}/{args.n_samples} done")

    (out / "dataset_meta.json").write_text(json.dumps(dict(
        n_samples=args.n_samples, grid_shape=[H, W], meters_per_px=mpp,
        downsample=f, sim_params=sim, seed=args.seed,
        channels=["wall_loss_db_per_crossing", "clutter_loss_db_per_m",
                  "log10_distance_m"],
        target="pathloss_db (float16), wall saturation applied"), indent=2))
    print(f"dataset written to {out}")


if __name__ == "__main__":
    main()
