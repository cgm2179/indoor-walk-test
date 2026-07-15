#!/usr/bin/env python3
"""
Step 1: Rasterize an architectural floor plan into a material-class grid
for indoor RF propagation simulation.

Input : floor plan PNG (clean, no annotations)
Output: material_grid.npy   (H x W uint8, one material id per pixel-cell)
        materials.json      (id -> name, penetration loss dB, color)
        transmitters.json   (Tx candidate coords extracted from green pins)
        preview_materials.png (color-coded overlay for visual QA)

Classification is classical CV (no ML): threshold -> morphology ->
thickness / density / shape heuristics. Deliberately tunable; all
magic numbers live in PARAMS.
"""
import cv2
import json
import numpy as np

PARAMS = dict(
    dark_thresh=110,        # gray < this  -> candidate structure (walls/columns/hatch)
    furn_lo=110, furn_hi=232,  # light-gray band -> furniture/fixtures
    open_kernel=3,          # opening removes strokes thinner than this
    thick_px=2.6,           # dist-transform radius (px): >= this -> "thick wall" (concrete)
    hatch_win=21,           # window for hatch-density (elevators/stairs cores)
    hatch_density=0.28,     # dark-pixel fraction -> hatched core
    column_area=(25, 900),  # solid blob area range for structural columns
    column_solidity=0.75,   # fill ratio for a blob to count as a solid column
)

# Material table (per-crossing penetration loss, dB).
# Values from the 2.4/5 GHz table in the project notes; edit freely.
MATERIALS = {
    0: dict(name="air",              loss_db=0.0,  color=(255, 255, 255)),
    1: dict(name="drywall_partition", loss_db=4.0,  color=(245, 166, 35)),
    2: dict(name="concrete_or_masonry", loss_db=15.0, color=(64, 64, 64)),
    3: dict(name="core_service_area", loss_db=20.0, color=(200, 30, 30)),   # elevators/stairs/WC: concrete+metal mix
    4: dict(name="furniture_clutter", loss_db=1.0,  color=(190, 210, 255)),
    5: dict(name="exterior_envelope", loss_db=15.0, color=(30, 60, 160)),   # set 3 dB if glass curtain wall
}

def main(in_path, out_prefix=""):
    bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # ---- 1. Transmitter pins (green markers) --------------------------------
    r, g, b = [rgb[:, :, i].astype(int) for i in range(3)]
    green = ((g > 120) & (g - r > 50) & (g - b > 50)).astype(np.uint8)
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(green, 8)
    tx = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 20:
            x, y, w, h = stats[i, :4]
            # pin tip = bottom-center of the marker glyph
            tx.append(dict(x=int(x + w / 2), y=int(y + h - 1)))
    # remove pin pixels so they don't pollute material masks
    pinmask = cv2.dilate(green, np.ones((5, 5), np.uint8))
    gray = gray.copy(); gray[pinmask > 0] = 255

    # ---- 2. Base masks -------------------------------------------------------
    dark = (gray < PARAMS["dark_thresh"]).astype(np.uint8)
    furn = ((gray >= PARAMS["furn_lo"]) & (gray <= PARAMS["furn_hi"])).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (PARAMS["open_kernel"],) * 2)
    structural = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)      # walls, columns, hatch
    thin_dark = cv2.subtract(dark, structural)                   # 1px dark strokes -> clutter

    # ---- 3. Hatched cores (elevator shafts / stairs) -------------------------
    dens = cv2.blur(dark.astype(np.float32), (PARAMS["hatch_win"],) * 2)
    hatched = ((dens > PARAMS["hatch_density"]) & (dark > 0)).astype(np.uint8)
    hatched = cv2.morphologyEx(hatched, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))

    # ---- 4. Thickness split: concrete vs drywall -----------------------------
    dist = cv2.distanceTransform(structural, cv2.DIST_L2, 3)
    thick_seed = (dist >= PARAMS["thick_px"]).astype(np.uint8)
    # grow seed back over the full stroke width
    thick = cv2.dilate(thick_seed, k) & structural

    # ---- 5. Solid columns -----------------------------------------------------
    filled = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k)
    n2, lbl2, st2, _ = cv2.connectedComponentsWithStats(filled, 8)
    columns = np.zeros_like(dark)
    lo, hi = PARAMS["column_area"]
    for i in range(1, n2):
        a = st2[i, cv2.CC_STAT_AREA]
        w, h = st2[i, cv2.CC_STAT_WIDTH], st2[i, cv2.CC_STAT_HEIGHT]
        if lo <= a <= hi and a / max(w * h, 1) >= PARAMS["column_solidity"] \
           and max(w, h) / max(min(w, h), 1) < 4:
            columns[lbl2 == i] = 1

    # ---- 5b. Exterior envelope: flood fill white bg from borders --------------
    ff = np.zeros((H + 2, W + 2), np.uint8)
    outside = np.zeros((H, W), np.uint8)
    openish = (gray > 200).astype(np.uint8)           # white-ish, floodable
    seeds = [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1), (W // 2, 0), (W // 2, H - 1)]
    flood_img = openish.copy()
    for sx, sy in seeds:
        if flood_img[sy, sx]:
            cv2.floodFill(flood_img, ff, (sx, sy), 2)
    outside = (flood_img == 2).astype(np.uint8)
    # exterior wall = structural pixels within reach of the outside region
    near_out = cv2.dilate(outside, np.ones((7, 7), np.uint8))
    exterior = (near_out & structural).astype(np.uint8)
    exterior = cv2.dilate(exterior, k) & structural   # cover full stroke width

    # ---- 6. Assemble grid (later assignments override earlier) ----------------
    grid = np.zeros((H, W), np.uint8)                 # 0 air
    grid[(furn > 0) | (thin_dark > 0)] = 4            # furniture / clutter
    grid[structural > 0] = 1                          # drywall by default
    grid[(thick > 0) | (columns > 0)] = 2             # concrete: thick walls + columns
    grid[hatched > 0] = 3                             # core service areas
    grid[exterior > 0] = 5                            # exterior envelope

    # ---- 7. Outputs -----------------------------------------------------------
    np.save(f"{out_prefix}material_grid.npy", grid)
    with open(f"{out_prefix}materials.json", "w") as f:
        json.dump({str(k_): dict(name=v["name"], loss_db=v["loss_db"])
                   for k_, v in MATERIALS.items()}, f, indent=2)
    with open(f"{out_prefix}transmitters.json", "w") as f:
        json.dump(dict(note="pixel coords; y down", transmitters=tx), f, indent=2)
    with open(f"{out_prefix}floorplan_meta.json", "w") as f:
        json.dump(dict(
            source=in_path, grid_shape=[H, W],
            scale_px_per_m=12.3,
            scale_status="ESTIMATE from furniture dimensions - confirm one real measurement",
            loss_semantics="loss_db is per WALL CROSSING: count contiguous runs of a "
                           "material along a ray as ONE crossing, not per cell",
            frequency_note="loss values ~2.4-5 GHz band; scale for 3.5 GHz NR if needed",
        ), f, indent=2)

    # preview: white bg, colored classes, Tx pins as green dots
    prev = np.full((H, W, 3), 255, np.uint8)
    for mid, m in MATERIALS.items():
        if mid == 0:
            continue
        prev[grid == mid] = m["color"]
    for t in tx:
        cv2.circle(prev, (t["x"], t["y"]), 7, (0, 160, 0), -1)
        cv2.circle(prev, (t["x"], t["y"]), 7, (0, 90, 0), 2)
    cv2.imwrite(f"{out_prefix}preview_materials.png",
                cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))

    # stats
    tot = H * W
    print(f"grid {W}x{H}  |  Tx pins: {tx}")
    for mid, m in MATERIALS.items():
        c = int((grid == mid).sum())
        print(f"  {mid} {m['name']:<20} {c:>8} px  {100*c/tot:5.2f}%")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit(
            "usage: python rasterize_floorplan.py <floorplan.png> [out_prefix]\n"
            "note: the committed STEP_1 outputs were generated from IMG_1863.png "
            "(752x1333), not the georeferenced 7th_Floor PNG (1150x515)."
        )
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
