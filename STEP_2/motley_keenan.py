#!/usr/bin/env python3
"""
Step 2: Log-distance + multi-wall (Motley-Keenan) path-loss heatmap.

    PL(cell) = FSPL(1 m, f) + 10 n log10(d / 1 m) + sum_m loss_m * crossings_m

For every grid cell, a straight ray is marched from the transmitter and the
wall crossings are counted per material -- a contiguous run of one material
along the ray counts ONCE (per the loss semantics in STEP_1), so wall
thickness in pixels does not inflate the loss.

Inputs : STEP_1/material_grid.npy, materials.json, transmitters.json,
         floorplan_meta.json
Outputs: pathloss_tx<i>.npy   (float32 dB, one per transmitter)
         heatmap_tx<i>.png    (received power map, matplotlib)
         heatmap_best_server.png
         sim_params.json

usage: python motley_keenan.py [--step1 STEP_1] [--out STEP_2]
                               [--freq-mhz 3500] [--n 3.0] [--eirp-dbm 23]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def consolidate_walls(grid, loss_db, gap_px=5, core_gap_px=9):
    """CAD walls are drawn as two parallel strokes with white space between;
    counting each stroke doubles every wall's loss. Close small same-material
    gaps so one physical wall = one crossing (and fragmented core blobs merge
    into solid banks). Only air/furniture cells are filled, corridors and
    rooms are far wider than the closing kernel. Narrow door openings may
    seal shut -- acceptable for a coverage model."""
    out = grid.copy()
    fillable = (grid == 0) | (grid == 4)
    wall_ids = [m for m in np.unique(grid) if m != 0 and loss_db[m] > 0]
    for m in sorted(wall_ids, key=lambda m: loss_db[m]):   # higher loss wins overlaps
        k = core_gap_px if m == 3 else gap_px
        closed = ndimage.binary_closing(grid == m, np.ones((k, k), bool))
        out[closed & fillable] = m
    return out


def compute_pathloss(grid, loss_db, loss_per_m, tx_xy, meters_per_px,
                     freq_mhz=3500.0, n_exp=2.0, wall_sat_db=60.0,
                     step_px=0.6, chunk_rows=8):
    """Motley-Keenan path loss (dB, float32) from tx_xy = (x_px, y_px) to
    every cell. loss_db / loss_per_m are length-256 lookup vectors indexed by
    material id: loss_db is charged once per contiguous crossing (walls),
    loss_per_m per meter of path inside the material (bulk clutter)."""
    H, W = grid.shape
    tx_x, tx_y = float(tx_xy[0]), float(tx_xy[1])

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    dist_px = np.hypot(gx - tx_x, gy - tx_y)
    max_dist = float(dist_px.max())
    K = int(np.ceil(max_dist / step_px)) + 1
    t = np.linspace(0.0, 1.0, K, dtype=np.float32)

    fspl_1m = 32.44 + 20 * np.log10(freq_mhz) - 60.0   # FSPL at d = 1 m, dB
    d_m = np.maximum(dist_px * meters_per_px, 1.0)      # clamp inside 1 m
    pl = (fspl_1m + 10.0 * n_exp * np.log10(d_m)).astype(np.float32)

    wall_ids = [m for m in np.unique(grid) if loss_db[m] > 0]
    bulk_ids = [m for m in np.unique(grid) if loss_per_m[m] > 0]
    for r0 in range(0, H, chunk_rows):
        r1 = min(r0 + chunk_rows, H)
        ex = gx[r0:r1].ravel()[:, None]                # ray endpoints, chunked
        ey = gy[r0:r1].ravel()[:, None]
        xi = np.clip((tx_x + t * (ex - tx_x)).round(), 0, W - 1).astype(np.int32)
        yi = np.clip((tx_y + t * (ey - tx_y)).round(), 0, H - 1).astype(np.int32)
        mats = grid[yi, xi]                            # (rays, K) uint8
        extra = np.zeros(mats.shape[0], np.float32)
        for m in wall_ids:
            hit = mats == m
            runs = hit[:, 0].astype(np.int32) + \
                (hit[:, 1:] & ~hit[:, :-1]).sum(axis=1, dtype=np.int32)
            extra += loss_db[m] * runs
        # per-ray sample spacing in meters (short rays are just oversampled)
        spacing_m = dist_px[r0:r1].ravel() / (K - 1) * meters_per_px
        for m in bulk_ids:
            extra += loss_per_m[m] * (mats == m).sum(axis=1) * spacing_m
        # Straight rays over-punish deep shadow: real energy reroutes through
        # corridors and around obstructions (COST-231 found multi-wall loss
        # grows sublinearly). Saturate total obstruction loss smoothly at
        # wall_sat_db; linear for small sums, capped for wall-stacks.
        extra = wall_sat_db * -np.expm1(-extra / wall_sat_db)
        pl[r0:r1] += extra.reshape(r1 - r0, W)
    return pl


def outside_mask(grid):
    """Cells outside the building envelope: air regions connected to the
    image border. Excluded from stats and grayed out in renders -- their
    rays cross the whole floor and mean nothing."""
    lbl, _ = ndimage.label(grid == 0)
    border = np.unique(np.concatenate(
        [lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]]))
    return np.isin(lbl, border[border > 0])


def render_heatmap(prx, grid, outside, mpp, title, out_png, vmin=-120, vmax=-40):
    H, W = grid.shape
    shown = np.where(outside, np.nan, prx)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad((0.93, 0.93, 0.93))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, W * mpp, H * mpp, 0])
    walls = np.isin(grid, [1, 2, 3, 5, 6, 7])
    overlay = np.zeros((H, W, 4))
    overlay[walls] = (0, 0, 0, 0.85)
    ax.imshow(overlay, extent=[0, W * mpp, H * mpp, 0])
    ax.set_xlabel("meters")
    ax.set_ylabel("meters")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="received power (dBm)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1", default="STEP_1")
    ap.add_argument("--out", default="STEP_2")
    ap.add_argument("--freq-mhz", type=float, default=3500.0,
                    help="carrier (default 3500 = NR n78)")
    ap.add_argument("--n", type=float, default=2.0, dest="n_exp",
                    help="path-loss exponent. Default 2.0 = free space, per "
                         "Motley-Keenan: walls are charged explicitly, so a "
                         "dense-office n~3 would double-count the environment. "
                         "Use n>2 only when dropping the wall terms.")
    ap.add_argument("--eirp-dbm", type=float, default=23.0)
    ap.add_argument("--wall-sat-db", type=float, default=60.0,
                    help="saturation of total obstruction loss (corridor "
                         "waveguiding proxy); set very large to disable")
    args = ap.parse_args()

    s1, out = Path(args.step1), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = np.load(s1 / "material_grid.npy")
    mats = json.loads((s1 / "materials.json").read_text())
    meta = json.loads((s1 / "floorplan_meta.json").read_text())
    txs = json.loads((s1 / "transmitters.json").read_text())["transmitters"]
    mpp = meta["meters_per_px"]

    loss_db = np.zeros(256, np.float32)
    loss_per_m = np.zeros(256, np.float32)
    for k, v in mats.items():
        loss_db[int(k)] = v["loss_db"]
        loss_per_m[int(k)] = v.get("loss_per_m_db", 0.0)

    grid = consolidate_walls(grid, loss_db)
    np.save(out / "material_grid_consolidated.npy", grid)
    outside = outside_mask(grid)
    np.save(out / "outside_mask.npy", outside)
    indoor = ~outside

    prx_all = []
    for i, tx in enumerate(txs, start=1):
        pl = compute_pathloss(grid, loss_db, loss_per_m,
                              (tx["x_px"], tx["y_px"]), mpp,
                              freq_mhz=args.freq_mhz, n_exp=args.n_exp,
                              wall_sat_db=args.wall_sat_db)
        np.save(out / f"pathloss_tx{i}.npy", pl)
        prx = args.eirp_dbm - pl
        prx_all.append(prx)
        render_heatmap(prx, grid, outside, mpp,
                       f"Tx{i} @ ({tx['x_m']:+.1f}, {tx['y_m']:+.1f}) m | "
                       f"{args.freq_mhz:.0f} MHz, n={args.n_exp}, "
                       f"EIRP {args.eirp_dbm:.0f} dBm",
                       out / f"heatmap_tx{i}.png")
        pi = prx[indoor]
        print(f"Tx{i} indoor Prx: median {np.median(pi):.1f} dBm, "
              f"p10 {np.percentile(pi, 10):.1f}, max {pi.max():.1f} | "
              f">= -100 dBm over {100 * (pi >= -100).mean():.1f}% of floor")

    if len(prx_all) > 1:
        best = np.maximum.reduce(prx_all)
        np.save(out / "prx_best_server.npy", best)
        render_heatmap(best, grid, outside, mpp,
                       f"Best server ({len(prx_all)} Tx) | "
                       f"{args.freq_mhz:.0f} MHz, n={args.n_exp}, "
                       f"EIRP {args.eirp_dbm:.0f} dBm",
                       out / "heatmap_best_server.png")
        bi = best[indoor]
        print(f"Best server indoor: median {np.median(bi):.1f} dBm, "
              f">= -100 dBm over {100 * (bi >= -100).mean():.1f}% of floor")

    (out / "sim_params.json").write_text(json.dumps(dict(
        model="log-distance + Motley-Keenan multi-wall",
        freq_mhz=args.freq_mhz, n_exp=args.n_exp, eirp_dbm=args.eirp_dbm,
        wall_sat_db=args.wall_sat_db,
        fspl_1m_db=round(32.44 + 20 * np.log10(args.freq_mhz) - 60.0, 2),
        step_px=0.6, meters_per_px=mpp,
        comment="pathloss_tx<i>.npy is PL in dB; received power = eirp_dbm - PL. "
                "Wall crossings count contiguous material runs once; furniture "
                "attenuates per meter of path. n=2 (free space) because walls "
                "are charged explicitly -- raising n as well double-counts. "
                "2D model: no reflections/diffraction/corridor waveguiding, so "
                "deep multi-wall shadows are pessimistic; no floor/ceiling paths.",
    ), indent=2))


if __name__ == "__main__":
    main()
