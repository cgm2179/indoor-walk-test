# STEP 1 — Floor Plan Rasterization (material-class grid)

Converts the georeferenced 7th-floor plan
(`7th_Floor_2nd_Indoor_Walk_Test_V2.2.png`) into a 2D grid where each
pixel-cell is a material class, plus transmitter candidate locations — all in
**float real-world coordinates** (meters / degrees) derived from the QGIS
ground control points. These are the inputs for STEP 2 (log-distance +
multi-wall / Motley-Keenan heatmap).

## Run it

```bash
python STEP_1/rasterize_floorplan.py 7th_Floor_2nd_Indoor_Walk_Test_V2.2.png --out STEP_1
```

Needs only `numpy`, `scipy`, `pillow` (no OpenCV). The georeference is read
from `<image>.TAB` automatically, or pass `--tab`.

## Files

| File | What it is |
|---|---|
| `rasterize_floorplan.py` | Classical-CV classifier (threshold → component-size → thickness/density heuristics). All tunables in `PARAMS` at the top. |
| `material_overrides.json` | Hand-labeled corrections for what the drawing can't show (glass vs drywall are identical thin lines). Applied automatically after classification. |
| `material_grid.npy` | 515 × 1150 `uint8` grid, one material id per cell (`np.load` it) |
| `materials.json` | material id → name + per-crossing penetration loss (dB) |
| `transmitters.json` | Tx pins in float px, local ENU meters, lon/lat, and EPSG:3857 |
| `floorplan_meta.json` | affine pixel→meters transforms, measured scale, GCP residuals |
| `preview_materials.png` | color-coded overlay for visual QA — check this first |

## Scale & coordinates (measured, float)

- **0.0679 m/px** — fit from the 3 QGIS ground control points in the `.TAB`
  file (similarity transform: uniform scale + rotation + translation). Floor
  plate ≈ **78.1 m × 35.0 m**.
- `floorplan_meta.json` carries three float affine transforms from pixel
  coords: → **local ENU meters** (use this for physical distances), →
  **EPSG:3857 pseudo-Mercator** (same frame as the walk-test CSV — note 3857
  lengths are inflated by ~1/cos(lat) ≈ 1.285, don't measure in it), and →
  **lon/lat**.
- Tx pins (extracted from the green markers, subpixel float):
  ~(636.3, 403.3) px and (663.0, 403.5) px → 1.8 m apart on the floor.

## Material classes / losses

Loss values reflect walked ground truth for this building (glass exterior and
lunch room, drywall-wrapped columns, soft/wood furniture, aluminum cubicle
panels), not generic assumptions.

| id | class | loss per crossing | assigned by |
|---|---|---|---|
| 0 | air | 0 dB | — |
| 1 | drywall partition (incl. columns, bathroom doors) | 4 dB | auto |
| 2 | concrete / masonry (thick shaft walls) | 15 dB | auto |
| 3 | core service area (elevators, stairs, WC) | 20 dB | auto |
| 4 | furniture, soft / wood (couches, chairs, desks) | 0.5 dB | auto |
| 5 | exterior glass curtain wall | 3 dB | auto |
| 6 | glass partition (lunch room, glass doors) | 2 dB | overrides only |
| 7 | cubicle aluminum panel | 6 dB | overrides only |

The auto-classifier cannot tell glass from drywall (both are thin lines in a
CAD drawing) — ids 6 and 7 exist so `material_overrides.json` can re-label
regions you know from walking the floor. Add an entry with a rough
`rect_px: [x0, y0, x1, y1]` (or `polygon_px`) and the ids it `applies_to`;
only those ids inside the box are re-labeled, so air and furniture are safe.
The seeded lunch-room box is a rough guess — adjust its extent.

Loss values are for the 2.4–5 GHz band; rescale for 3.5 GHz NR if needed.
`loss_db` is per **wall crossing** — a contiguous run of one material along a
ray counts once, not per cell. Note: if the exterior glazing is low-E
(metal-coated, common in modern offices), outdoor↔indoor loss is far higher
than 3 dB (20+ dB) — irrelevant for indoor-only simulation, but it matters if
you ever model street-level donors.

## Known limitations

- **GCP quality**: the 3 control points are nearly collinear and mutually
  inconsistent at the few-meter level (fit residuals 1.2–2.4 m, see
  `gcp_residuals_m` in the meta). Adding a 4th well-spread GCP in QGIS and
  re-running would tighten the scale.
- **Classification**: some dense wall junctions over-fire as "core service
  area" (red in the preview). Tune `PARAMS` (esp. `hatch_density`,
  `thick_radius_px`) and re-run; the preview makes misfires easy to spot.
- The earlier v1 outputs (built from the off-repo photo `IMG_1863.png`) are
  preserved in git history at commit `bb3de5a`.
