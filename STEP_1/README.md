# STEP 1 — Floor Plan Rasterization (material-class grid)

Converts the 7th-floor plan image into a 2D grid where each pixel-cell is a
material class, plus the transmitter candidate locations. These are the inputs
for STEP 2 (log-distance + multi-wall / Motley-Keenan heatmap).

## Files

| File | What it is |
|---|---|
| `rasterize_floorplan.py` | Classical-CV classifier (threshold → morphology → thickness/density/shape heuristics). All tunables are in `PARAMS` at the top. Run: `python rasterize_floorplan.py <floorplan.png> [out_prefix]` |
| `material_grid.npy` | 752 × 1333 `uint8` grid, one material id per cell (`np.load` it) |
| `materials.json` | material id → name + per-crossing penetration loss (dB) |
| `transmitters.json` | Tx candidates extracted from the two green pins: (636, 523) and (663, 523), pixel coords, y-down |
| `floorplan_meta.json` | grid shape, px-per-meter scale, loss semantics notes |
| `preview_materials.png` | color-coded overlay for visual QA — check this first |

## Material classes / losses

| id | class | loss per crossing |
|---|---|---|
| 0 | air | 0 dB |
| 1 | drywall partition | 4 dB |
| 2 | concrete / masonry (thick walls, columns) | 15 dB |
| 3 | core service area (elevators, stairs, WC) | 20 dB |
| 4 | furniture / clutter | 1 dB |
| 5 | exterior envelope | 15 dB (use ~3 dB if glass curtain wall) |

Loss values are for the 2.4–5 GHz band; rescale for 3.5 GHz NR if needed.

## Caveats before STEP 2

- **Source image**: the grid was generated from `IMG_1863.png` (752 × 1333, the
  clean floor plan photo uploaded to the claude.ai session), **not** from
  `7th_Floor_2nd_Indoor_Walk_Test_V2.2.png` (1150 × 515, the georeferenced
  export in the repo root). Keep a copy of the source image alongside this
  folder if you want to re-run or re-tune the classifier.
- **Scale**: `scale_px_per_m = 12.3` is an *estimate* from furniture
  dimensions. Confirm with one real-world measurement (e.g. a known corridor
  width) before trusting distances in the path-loss model.
- **Loss semantics**: `loss_db` is per **wall crossing** — a contiguous run of
  one material along a ray counts once, not per cell.
