#!/usr/bin/env python3
"""
Phase A v2 — build the 7-class model grid + manifest_v2.json.

The one grid change v2 needs: the interior lunch-room / meeting-room glass
(ITU-R P.2040 ordinary glass, ~3 dB) must separate from the exterior low-E
facade (measured 20 dB). v1 folded both into class 5; here they become
classes 5 (exterior low-E) and 6 (interior glass), so the surrogate input
grows to IN_CH = 10 (7 one-hot + tx + freq + dist).

The 7-class model grid is re-pooled from STEP_2/material_grid_consolidated.npy
(which already carries interior glass as class 6, 4599 px) using the SAME
max-loss / majority pooling as v1 phase_a.prepare_grid, only with a 7-class
fold and the v2 loss ordering (exterior low-E now dominates mixed perimeter
blocks, which is correct). Geometry (inside/walkable) is identical to v1.

Writes into --out (default "SIM V2"): grid_model_v2.npy, inside_mask_v2.npy,
walkable_mask_v2.npy, manifest_v2.json.

usage:
  python "SIM V2/phase_a_v2.py" --prepare --repo . --out "SIM V2"
  python "SIM V2/phase_a_v2.py" --test
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

H_MODEL, W_MODEL = 256, 448

# 8-class STEP_2 -> 7-class model. Only difference from v1's FOLD_8_TO_6:
# glass_partition (6) stays 6 (interior glass) instead of folding into 5.
FOLD_8_TO_7 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 4}

# Ordering weights for max-loss pooling of mixed blocks. These decide which
# material "wins" a block that straddles two classes; they are NOT the physics
# losses (those live in physics_v2.MATERIALS7). Exterior low-E (5) is heaviest
# so facade blocks stay facade; interior glass (6) sits near drywall.
POOL_WEIGHT = np.array([0.0, 4.0, 15.0, 20.0, 0.0, 25.0, 4.5], np.float32)


def prepare(repo: Path, out: Path):
    src = np.load(repo / "STEP_2" / "material_grid_consolidated.npy")
    outside = np.load(repo / "STEP_2" / "outside_mask.npy")
    meta1 = json.loads((repo / "STEP_1" / "floorplan_meta.json").read_text())
    mpp0 = meta1["meters_per_px"]

    fold = np.zeros(256, np.uint8)
    for k, v in FOLD_8_TO_7.items():
        fold[k] = v
    src7 = fold[src]

    Hs, Ws = src7.shape
    cell = (Ws * mpp0) / W_MODEL                    # square cells (~0.1744 m)
    rows = int(round(Hs * mpp0 / cell))
    pad_top = (H_MODEL - rows) // 2

    grid = np.zeros((H_MODEL, W_MODEL), np.uint8)   # padding = air
    inside = np.zeros((H_MODEL, W_MODEL), bool)
    ye = np.linspace(0, Hs, rows + 1)
    xe = np.linspace(0, Ws, W_MODEL + 1)
    for i in range(rows):
        y0, y1 = int(ye[i]), max(int(ye[i + 1]), int(ye[i]) + 1)
        for j in range(W_MODEL):
            x0, x1 = int(xe[j]), max(int(xe[j + 1]), int(xe[j]) + 1)
            blk = src7[y0:y1, x0:x1].ravel()
            w = POOL_WEIGHT[blk]
            if w.max() > 0:                          # a wall class present
                grid[pad_top + i, j] = blk[int(np.argmax(w))]
            elif (blk == 4).mean() >= 0.5:           # majority furniture
                grid[pad_top + i, j] = 4
            inside[pad_top + i, j] = (~outside[y0:y1, x0:x1]).mean() >= 0.5

    # Core-fragment cleanup, identical to v1 (small class-3 misfires -> drywall)
    lbl, n = ndimage.label(grid == 3)
    if n:
        areas = ndimage.sum_labels(np.ones_like(lbl), lbl, np.arange(1, n + 1))
        grid[np.isin(lbl, np.nonzero(areas < 40)[0] + 1)] = 1

    walkable = np.isin(grid, [0, 4]) & inside        # Rule R8

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "grid_model_v2.npy", grid)
    np.save(out / "inside_mask_v2.npy", inside)
    np.save(out / "walkable_mask_v2.npy", walkable)

    write_manifest(grid, walkable, inside, cell, mpp0, out, repo)

    hist = {int(k): int(v) for k, v in zip(*np.unique(grid, return_counts=True))}
    print(f"grid {grid.shape} cell {cell:.4f} m | classes {hist}")
    print(f"  interior glass (id 6): {hist.get(6, 0)} cells "
          f"{'(EMPTY - add an id-6 override; see MODEL_CARD_V2 5.2)' if not hist.get(6) else 'OK'}")
    print(f"  walkable {int(walkable.sum())} cells | wrote grid/masks/manifest to {out}")
    return grid, inside, walkable, cell


def write_manifest(grid, walkable, inside, cell, mpp0, out, repo):
    # v2 physics table + frequency union, imported from the engine
    sys.path.insert(0, str(repo / "SIM"))
    import physics_v2 as P
    rows_present = np.nonzero(inside.any(1))[0]
    manifest = dict(
        version="sim-v2.0",
        grid_shape=[H_MODEL, W_MODEL],
        cell_size_m=round(float(cell), 6),
        floor_rows=[int(rows_present[0]), int(rows_present[-1] + 1)],
        n_classes=7,
        channels=["onehot_air", "onehot_drywall", "onehot_concrete",
                  "onehot_core", "onehot_furniture", "onehot_exterior_lowE",
                  "onehot_interior_glass", "tx_gaussian_sigma2",
                  "freq_feature", "log10_distance_m_over3"],
        in_ch=10,
        tx_blob_sigma_cells=2.0,
        dist_channel_norm=3.0,
        # widened clip: frequency-scaled 5.5/6 GHz losses are genuinely large
        # (indoor p99.9 ~300 dB at 6 GHz), so [40,170] would clip ~half the map
        norm=dict(pl_min_db=40.0, pl_range_db=190.0,
                  freq_log_lo_mhz=619.0, freq_log_hi_mhz=6125.0),
        physics=dict(model="enhanced_motley_keenan_v2",
                     source="ITU-R P.2040 permittivity + Fresnel slab (angle+"
                            "thickness) + low-E resistive sheet + UTD diffraction",
                     n_exp=2.0, d0_m=1.0, saturation="none (UTD supplies shadow)",
                     doc="SIM V2/v2_physics.pdf"),
        freqs_mhz=list(P.FREQS_MHZ_V2),
        materials=[dict(id=m["id"], name=m["name"],
                        p2040=m.get("p2040"), t_ref_m=m.get("t_ref_m"),
                        r_sheet_ohm_sq=m.get("r_sheet_ohm_sq"),
                        per_metre=bool(m.get("per_metre")),
                        color=m.get("color")) for m in P.MATERIALS7],
        scale_status="literature (P.2040 + fitted sub-resolution excess); "
                     "NOT yet calibrated to this building (Phase D fits it)",
        grid_sha256=hashlib.sha256(grid.tobytes()).hexdigest()[:16],
        walkable_mask_sha256=hashlib.sha256(walkable.tobytes()).hexdigest()[:16],
    )
    (out / "manifest_v2.json").write_text(json.dumps(manifest, indent=2))


def run_test(repo: Path):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        grid, inside, walkable, cell = prepare(repo, Path(td))
        assert grid.shape == (H_MODEL, W_MODEL)
        cls = set(np.unique(grid).tolist())
        assert cls <= set(range(7)), f"unexpected classes {cls}"
        assert 5 in cls, "exterior glass (5) missing"
        assert walkable.sum() > 10000, "too few walkable cells"
        man = json.loads((Path(td) / "manifest_v2.json").read_text())
        assert man["n_classes"] == 7 and man["in_ch"] == 10
        assert man["norm"]["pl_range_db"] == 190.0
        assert len(man["freqs_mhz"]) == 9
        print("phase_a_v2 tests PASSED "
              f"(classes {sorted(cls)}, walkable {int(walkable.sum())})")
        if 6 not in cls:
            print("  WARNING: interior glass (id 6) is empty in the model grid.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--repo", default=".", help="repo root")
    ap.add_argument("--out", default="SIM V2")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    if a.test:
        run_test(repo)
    if a.prepare:
        prepare(repo, Path(a.out))
