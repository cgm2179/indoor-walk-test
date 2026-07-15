#!/usr/bin/env python3
"""
Step 1 (v2): Rasterize the georeferenced floor plan into a material-class grid
for indoor RF propagation simulation.

Input : floor plan PNG + MapInfo .TAB georeference (3 GCPs: pixel <-> lon/lat)
Output: material_grid.npy     (H x W uint8, one material id per pixel-cell)
        materials.json        (id -> name, penetration loss dB)
        transmitters.json     (Tx pins in float px, local meters, lon/lat, EPSG:3857)
        floorplan_meta.json   (affine pixel->meters transforms, float scale)
        preview_materials.png (color-coded overlay for visual QA)

All coordinates and scales are floats in real units (meters / degrees) derived
from the QGIS ground control points -- no integer-pixel scale estimates.

Classification is classical CV (no ML) on numpy/scipy only (no OpenCV):
threshold -> morphology -> thickness / density / shape heuristics.
All magic numbers live in PARAMS.

usage: python rasterize_floorplan.py <floorplan.png> [--tab <georef.TAB>] [--out <dir>]
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

PARAMS = dict(
    dark_thresh=120,        # gray < this -> structure candidate (walls/columns/hatch)
    furn_lo=150, furn_hi=240,  # light-gray band -> furniture/fixtures
    min_struct_area=50,     # dark blobs smaller than this are text/clutter, not walls
    thick_radius_px=2.0,    # dist-transform radius: >= this -> "thick wall" (concrete)
    hatch_win=15,           # window for hatch density (elevator/stair cores)
    hatch_density=0.28,     # dark-pixel fraction in window -> hatched core
    column_area=(15, 900),  # isolated solid blob area range for structural columns
    column_solidity=0.60,   # bbox fill ratio for a blob to count as a solid column
    column_aspect=4.0,      # max side ratio for a column blob
    open_bg_thresh=200,     # gray > this is floodable background for outside detection
    exterior_reach_px=7,    # structure within this distance of "outside" = envelope
    pin_grow_px=2,          # dilation of green-pin mask before erasing pins
)

# Material table (per-crossing penetration loss, dB), ~2.4-5 GHz band.
# Loss values reflect walked ground truth for this building: exterior and
# lunch-room enclosure are glass, columns are drywall-wrapped, furniture is
# soft/wood, cubicle partitions are aluminum-skinned. Ids 6 and 7 are never
# assigned by the auto-classifier (glass and drywall are indistinguishable
# thin lines in the drawing) -- they come from material_overrides.json.
MATERIALS = {
    0: dict(name="air",                 loss_db=0.0,  color=(255, 255, 255)),
    1: dict(name="drywall_partition",   loss_db=4.0,  color=(245, 166, 35)),
    2: dict(name="concrete_or_masonry", loss_db=15.0, color=(64, 64, 64)),
    3: dict(name="core_service_area",   loss_db=20.0, color=(200, 30, 30)),
    4: dict(name="furniture_soft_wood", loss_db=0.5,  color=(190, 210, 255)),
    5: dict(name="exterior_glass_curtain_wall", loss_db=3.0, color=(30, 60, 160)),
    6: dict(name="glass_partition",     loss_db=2.0,  color=(0, 185, 205)),
    7: dict(name="cubicle_aluminum_panel", loss_db=6.0, color=(150, 150, 165)),
}

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


# --------------------------------------------------------------------------
# Georeference: MapInfo TAB GCPs -> affine pixel->meters transforms
# --------------------------------------------------------------------------
def parse_tab_gcps(tab_path):
    """Return [(lon, lat, px_x, px_y, label), ...] from a MapInfo raster TAB."""
    pat = re.compile(
        r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)\s*"
        r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)\s*Label\s+\"([^\"]*)\"")
    gcps = []
    for m in pat.finditer(Path(tab_path).read_text()):
        lon, lat, px, py, label = m.groups()
        gcps.append((float(lon), float(lat), float(px), float(py), label))
    if len(gcps) < 3:
        raise ValueError(f"need >= 3 GCPs in {tab_path}, found {len(gcps)}")
    return gcps


def lonlat_to_local_m(lon, lat, lon0, lat0):
    """Equirectangular ENU meters around (lon0, lat0), WGS84 curvature radii.
    Accurate to ~mm over a building footprint."""
    phi = math.radians(lat0)
    s2 = math.sin(phi) ** 2
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * s2)            # prime vertical radius
    M = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * s2) ** 1.5  # meridional radius
    x = math.radians(lon - lon0) * N * math.cos(phi)
    y = math.radians(lat - lat0) * M
    return x, y


def lonlat_to_mercator(lon, lat):
    """EPSG:3857 pseudo-Mercator (matches the walk-test CSV convention)."""
    x = WGS84_A * math.radians(lon)
    y = WGS84_A * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def fit_similarity(px_xy, world_xy):
    """Least-squares similarity (uniform scale + rotation + translation) from
    y-down pixel coords to y-up world meters. A full affine is NOT used here:
    the three GCPs are nearly collinear, which makes a 6-parameter fit
    ill-conditioned; a similarity stays stable. Returns 2x3 matrix."""
    q = np.asarray(px_xy, float) * [1.0, -1.0]   # flip to y-up
    w = np.asarray(world_xy, float)
    qc, wc = q - q.mean(0), w - w.mean(0)
    denom = (qc ** 2).sum()
    a = (qc * wc).sum() / denom                       # s*cos(theta)
    b = (qc[:, 0] * wc[:, 1] - qc[:, 1] * wc[:, 0]).sum() / denom  # s*sin(theta)
    R = np.array([[a, -b], [b, a]])
    t = w.mean(0) - R @ q.mean(0)
    M = np.column_stack([R * [1.0, -1.0], t])     # fold the y-flip back in
    scale = float(np.hypot(a, b))
    rot = math.degrees(math.atan2(b, a))
    resid = [float(np.hypot(*(apply_affine(M, x, y) - np.asarray(wi))))
             for (x, y), wi in zip(px_xy, world_xy)]
    return M, scale, rot, resid


def apply_affine(M, x, y):
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2],
            M[1, 0] * x + M[1, 1] * y + M[1, 2])


def build_georef(tab_path):
    gcps = parse_tab_gcps(tab_path)
    lon0 = float(np.mean([g[0] for g in gcps]))
    lat0 = float(np.mean([g[1] for g in gcps]))
    px = [(g[2], g[3]) for g in gcps]
    local = [lonlat_to_local_m(g[0], g[1], lon0, lat0) for g in gcps]

    M_local, scale, rot, resid = fit_similarity(px, local)

    # Mercator is conformal: locally the same similarity scaled by 1/cos(lat).
    # Deriving it from M_local (instead of a separate fit) keeps the two
    # transforms consistent with each other.
    k = 1.0 / math.cos(math.radians(lat0))
    mx0, my0 = lonlat_to_mercator(lon0, lat0)
    M_merc = np.array([[M_local[0, 0] * k, M_local[0, 1] * k, M_local[0, 2] * k + mx0],
                       [M_local[1, 0] * k, M_local[1, 1] * k, M_local[1, 2] * k + my0]])

    # lon/lat from local meters via the same curvature radii used forward
    phi = math.radians(lat0)
    s2 = math.sin(phi) ** 2
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * s2)
    Mrad = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * s2) ** 1.5
    dlon = math.degrees(1.0) / (N * math.cos(phi))   # deg east per local meter
    dlat = math.degrees(1.0) / Mrad                  # deg north per local meter
    M_lonlat = np.array([
        [M_local[0, 0] * dlon, M_local[0, 1] * dlon, M_local[0, 2] * dlon + lon0],
        [M_local[1, 0] * dlat, M_local[1, 1] * dlat, M_local[1, 2] * dlat + lat0]])

    return dict(
        gcps=[dict(label=g[4], px=[g[2], g[3]], lon=g[0], lat=g[1]) for g in gcps],
        origin_lonlat=[lon0, lat0],
        affine_px_to_local_m=M_local.tolist(),
        affine_px_to_mercator=M_merc.tolist(),
        affine_px_to_lonlat=M_lonlat.tolist(),
        meters_per_px=scale,
        map_rotation_deg=rot,
        gcp_residuals_m=[round(r_, 2) for r_ in resid],
        gcp_residuals_comment="Distance (m) between each hand-placed GCP and the "
                 "fitted transform, i.e. GPS-grade placement error. 1-2 m is FINE "
                 "for this project: room-to-room geometry comes from the drawing "
                 "itself (pixel-exact), so the residuals only affect the absolute "
                 "geo-anchor and the overall scale by a few percent -- well under "
                 "1 dB in the 10*n*log10(d) path-loss term. To tighten: add a 4th "
                 "GCP in QGIS placed off the line of the current three (they are "
                 "nearly collinear, which is also why a similarity fit is used "
                 "instead of a full affine).",
    ), M_local, M_merc, M_lonlat


# --------------------------------------------------------------------------
# Material classification
# --------------------------------------------------------------------------
def square(k):
    return np.ones((k, k), bool)


def classify(rgb, p=PARAMS):
    gray = rgb.astype(float).mean(axis=2)
    H, W = gray.shape

    # ---- 1. Transmitter pins (green markers): locate tips, then erase -------
    r, g, b = (rgb[:, :, i].astype(int) for i in range(3))
    green = (g > 120) & (g - r > 50) & (g - b > 50)
    lbl, n = ndimage.label(green)
    tx = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lbl == i)
        if len(ys) < 20:
            continue
        # pin tip = subpixel centroid of the glyph's bottom two rows
        bottom = ys >= ys.max() - 1
        tx.append(dict(x_px=float(xs[bottom].mean()), y_px=float(ys[bottom].mean())))
    pinmask = ndimage.binary_dilation(green, square(2 * p["pin_grow_px"] + 1))
    gray = gray.copy()
    gray[pinmask] = 255.0

    # ---- 2. Base masks -------------------------------------------------------
    dark = gray < p["dark_thresh"]
    furn = (gray >= p["furn_lo"]) & (gray <= p["furn_hi"])

    # ---- 3. Structure vs text/specks: connected-component size ---------------
    # Walls form large connected networks; room labels and dimension text are
    # small isolated blobs. Size beats morphological opening here because the
    # interior walls are only ~2 px wide and would not survive a 3x3 opening.
    lbl_d, n_d = ndimage.label(dark, structure=square(3))
    areas = ndimage.sum_labels(np.ones_like(lbl_d), lbl_d, index=np.arange(1, n_d + 1))
    big = np.zeros(n_d + 1, bool)
    big[1:] = areas >= p["min_struct_area"]
    structural = big[lbl_d]
    small_dark = dark & ~structural

    # ---- 4. Isolated solid columns among the small blobs ---------------------
    columns = np.zeros_like(dark)
    lo, hi = p["column_area"]
    for i, sl in enumerate(ndimage.find_objects(lbl_d), start=1):
        if big[i] or sl is None:
            continue
        blob = lbl_d[sl] == i
        a = blob.sum()
        h, w = blob.shape
        if lo <= a <= hi and a / (h * w) >= p["column_solidity"] \
           and max(h, w) / max(min(h, w), 1) < p["column_aspect"]:
            columns[sl] |= blob

    # ---- 5. Thickness split: concrete vs drywall ------------------------------
    dist = ndimage.distance_transform_edt(structural)
    thick = ndimage.binary_dilation(dist >= p["thick_radius_px"], square(3)) & structural

    # ---- 6. Hatched cores (elevator shafts / stairs / WC) --------------------
    dens = ndimage.uniform_filter(dark.astype(float), size=p["hatch_win"])
    hatched = (dens > p["hatch_density"]) & dark
    hatched = ndimage.binary_closing(hatched, square(9))

    # ---- 7. Exterior envelope: flood the background from image borders -------
    openish = gray > p["open_bg_thresh"]
    lbl_o, _ = ndimage.label(openish)
    border_labels = np.unique(np.concatenate([
        lbl_o[0, :], lbl_o[-1, :], lbl_o[:, 0], lbl_o[:, -1]]))
    outside = np.isin(lbl_o, border_labels[border_labels > 0])
    near_out = ndimage.binary_dilation(outside, square(2 * p["exterior_reach_px"] + 1))
    exterior = ndimage.binary_dilation(near_out & structural, square(3)) & structural

    # ---- 8. Assemble grid (later assignments override earlier) ----------------
    grid = np.zeros((H, W), np.uint8)                  # 0 air
    grid[furn | small_dark] = 4                        # furniture / clutter / text
    grid[structural | columns] = 1                     # drywall (columns are
                                                       #   drywall-wrapped here)
    grid[thick] = 2                                    # concrete: thick shaft walls
    grid[hatched & structural] = 3                     # core service areas
    grid[exterior] = 5                                 # exterior envelope (glass)
    return grid, tx


def apply_overrides(grid, overrides_path):
    """Hand-labeled material corrections for what the drawing can't show
    (glass vs drywall look identical as lines). Each entry re-labels only the
    ids in applies_to inside its region, so rough boxes are safe: air and
    furniture stay untouched unless explicitly listed."""
    spec = json.loads(Path(overrides_path).read_text())
    H, W = grid.shape
    for ov in spec.get("overrides", []):
        region = np.zeros((H, W), bool)
        if "rect_px" in ov:
            x0, y0, x1, y1 = ov["rect_px"]
            region[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = True
        elif "polygon_px" in ov:
            from matplotlib.path import Path as MplPath
            yy, xx = np.mgrid[0:H, 0:W]
            pts = np.column_stack([xx.ravel(), yy.ravel()])
            region = MplPath(ov["polygon_px"]).contains_points(pts).reshape(H, W)
        mask = region & np.isin(grid, ov.get("applies_to", [1, 2, 3, 5]))
        grid[mask] = ov["material"]
        print(f"  override '{ov.get('label', '?')}': "
              f"{int(mask.sum())} px -> {MATERIALS[ov['material']]['name']}")
    return grid


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("image", help="floor plan PNG")
    ap.add_argument("--tab", help="MapInfo .TAB georeference (defaults to <image>.TAB)")
    ap.add_argument("--out", default=".", help="output directory (default: cwd)")
    ap.add_argument("--overrides",
                    help="material overrides JSON (defaults to "
                         "<out>/material_overrides.json when present)")
    args = ap.parse_args()

    img_path = Path(args.image)
    tab_path = Path(args.tab) if args.tab else img_path.with_suffix(".TAB")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    grid, tx = classify(rgb)
    H, W = grid.shape

    ov_path = Path(args.overrides) if args.overrides else out / "material_overrides.json"
    if ov_path.exists():
        grid = apply_overrides(grid, ov_path)

    geo, M_local, M_merc, M_lonlat = build_georef(tab_path)
    for t in tx:
        t["x_m"], t["y_m"] = apply_affine(M_local, t["x_px"], t["y_px"])
        t["lon"], t["lat"] = apply_affine(M_lonlat, t["x_px"], t["y_px"])
        t["mercator_x"], t["mercator_y"] = apply_affine(M_merc, t["x_px"], t["y_px"])

    np.save(out / "material_grid.npy", grid)
    (out / "materials.json").write_text(json.dumps(
        {str(k): dict(name=v["name"], loss_db=v["loss_db"])
         for k, v in MATERIALS.items()}, indent=2))
    (out / "transmitters.json").write_text(json.dumps(dict(
        note="x_px/y_px: float pixel coords, y down. x_m/y_m: local ENU meters "
             "(east/north) about origin_lonlat in floorplan_meta.json. "
             "mercator_*: EPSG:3857, same frame as the walk-test CSV.",
        transmitters=tx), indent=2))
    (out / "floorplan_meta.json").write_text(json.dumps(dict(
        source=img_path.name,
        georeference=tab_path.name,
        grid_shape=[H, W],
        crs_note="local ENU meters via WGS84 equirectangular about origin_lonlat; "
                 "EPSG:3857 lengths are inflated by ~1/cos(lat) (~1.285 here) -- "
                 "use the local-meter affine for physical distances",
        scale_status="measured from the 3 QGIS ground control points in the .TAB "
                     "(float meters, no integer-pixel estimate)",
        loss_semantics="loss_db is per WALL CROSSING: count contiguous runs of a "
                       "material along a ray as ONE crossing, not per cell",
        frequency_note="loss values ~2.4-5 GHz band; scale for 3.5 GHz NR if needed",
        **geo), indent=2))

    # preview: white bg, colored classes, Tx pins as green dots
    prev = np.full((H, W, 3), 255, np.uint8)
    for mid, m in MATERIALS.items():
        if mid:
            prev[grid == mid] = m["color"]
    yy, xx = np.mgrid[0:H, 0:W]
    for t in tx:
        d2 = (xx - t["x_px"]) ** 2 + (yy - t["y_px"]) ** 2
        prev[d2 <= 49] = (0, 160, 0)
        prev[(d2 > 49) & (d2 <= 81)] = (0, 90, 0)
    Image.fromarray(prev).save(out / "preview_materials.png")

    # stats
    mpp = geo["meters_per_px"]
    print(f"grid {W}x{H} | {mpp:.4f} m/px ({W * mpp:.1f} x {H * mpp:.1f} m) | "
          f"GCP residuals {geo['gcp_residuals_m']} m")
    for t in tx:
        print(f"  Tx px=({t['x_px']:.1f}, {t['y_px']:.1f})  "
              f"local=({t['x_m']:+.2f}, {t['y_m']:+.2f}) m  "
              f"lon/lat=({t['lon']:.7f}, {t['lat']:.7f})")
    tot = H * W
    for mid, m in MATERIALS.items():
        c = int((grid == mid).sum())
        print(f"  {mid} {m['name']:<20} {c:>8} px  {100 * c / tot:5.2f}%")


if __name__ == "__main__":
    main()
