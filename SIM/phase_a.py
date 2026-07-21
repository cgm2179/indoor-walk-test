#!/usr/bin/env python3
"""
Phase A — Ground-truth physics generator (multi-wall / Motley-Keenan).

Spec: "UNet Path-Loss Surrogate — Design Context & Build Instructions" v1.0.
Implements A.1 (grid prep at 256x448), A.2 (indoor point source, rules R3/R8),
A.3 (outdoor base-station plane-wave mode), the arrival-time maps of F.5, the
manifest (R6), and the A.4 unit tests.

Deviations from the spec text, by design (documented in manifest notes):
- Source grid is the georeferenced 515x1150 raster (not the older 752x1333);
  the scale is MEASURED from QGIS ground control points (0.0679 m/px), which
  satisfies the "confirm scale" blocking item with something better than the
  12.3 px/m furniture estimate.
- The floor's true aspect (78.1 x 35.0 m) is wider than 448:256; the floor
  occupies 448 x 201 cells and is vertically centered with padding rows
  (outside, non-walkable). Cells stay square: cell_size_m = 78.12/448.
- Resampling uses max-loss block pooling instead of plain nearest-neighbor:
  labels stay integers (the spec's stated reason for NN) AND 2-px walls
  survive the 2.57x stride, which plain NN would randomly delete.

usage:
  python SIM/phase_a.py --test        # A.4 unit tests
  python SIM/phase_a.py --prepare    # write grid_model.npy, walkable, manifest
  python SIM/phase_a.py --sample    # demo indoor + BS maps + timing
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "SIM"

H_MODEL, W_MODEL = 256, 448
D0_M = 1.0
N_EXP = 2.0
C_MPS = 299_792_458.0
STEP_CELL = 0.5              # ray sampling step, in cells

# 6-class table (spec §3 A.1, class 5 = glass curtain wall per walked ground
# truth). Walls: loss_db per contiguous crossing (Rule R3). Furniture is the
# one deviation: at 0.17 m cells an open-plan ray crosses ~50 separate
# furniture runs, so the spec's 1 dB/crossing explodes to +50 dB; it uses the
# spec's own 12.3-1 Beer-Lambert form instead (dB per meter of path), pulled
# forward to v1 for this class only.
MATERIALS6 = [
    dict(id=0, name="air",               loss_db=0.0,  loss_per_m_db=0.0, color="#ffffff"),
    dict(id=1, name="drywall_partition", loss_db=4.0,  loss_per_m_db=0.0, color="#f5a623"),
    dict(id=2, name="concrete_masonry",  loss_db=15.0, loss_per_m_db=0.0, color="#404040"),
    dict(id=3, name="core_service_area", loss_db=20.0, loss_per_m_db=0.0, color="#c81e1e"),
    dict(id=4, name="furniture_clutter", loss_db=0.0,  loss_per_m_db=0.3, color="#bed2ff"),
    dict(id=5, name="exterior_glass",    loss_db=3.0,  loss_per_m_db=0.0, color="#1e3ca0"),
]
FOLD_8_TO_6 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5, 7: 4}

# B.1 frequency plan and (crude, monotone) material-loss multipliers;
# replaced by Phase D calibration.
FREQS_MHZ = [2442.0, 3500.0, 5500.0, 6125.0]
FREQ_LOSS_MULT = {2442.0: 1.00, 3500.0: 1.15, 5500.0: 1.30, 6125.0: 1.40}

# ---- physics v1.2, QUEUED (activate by flag + regenerate dataset + retrain)
# Per-material absolute dB per crossing, from measured attenuation tables
# (2.4 / 5 GHz anchors; 3.5 and 6.125 GHz log-f interpolated/extrapolated).
# Measurements show concrete and glass roughly DOUBLE from 2.4 to 5 GHz while
# drywall grows slower -- the v1.1 global multiplier under-scales them.
# NOTE: exterior assumes STANDARD glass; if the curtain wall is low-E/coated,
# raise class 5 to ~10/12/15/16 dB.
PHYSICS_VARIANT = "v1.1"          # set "v1.2" -> make dataset -> retrain
LOSS_DB_V2 = {                    # id: {f_mhz: dB per crossing}
    1: {2442: 3.0,  3500: 5.3,  5500: 8.0,  6125: 8.7},    # drywall
    2: {2442: 15.0, 3500: 22.7, 5500: 31.5, 6125: 34.1},   # concrete
    3: {2442: 22.0, 3500: 28.7, 5500: 36.5, 6125: 38.9},   # core (concrete+metal)
    5: {2442: 3.0,  3500: 4.5,  5500: 6.3,  6125: 6.8},    # standard glass
}
LOSS_PER_M_V2 = {4: {2442: 0.30, 3500: 0.45, 5500: 0.62, 6125: 0.68}}  # clutter

# ---- physics v2.0 config, SCAFFOLD ONLY (see MODEL_CARD_v2.md) --------------
# Enhanced / modified Motley-Keenan (IEEE Xplore doc 8016211). This block is
# DATA describing the v2 model; the engine functions that consume angle,
# thickness, and the frequency exponent are NOT implemented yet (they are the
# workflow in MODEL_CARD_v2.md §11 steps 3-4). loss_table() ignores this
# unless PHYSICS_VARIANT == "v2.0", which nothing sets today, so the deployed
# v1.1 model is untouched.
#
# Key v2 material change (MEASURED, not assumed): the FCC HQ facade is low-E /
# metal-coated glass -> exterior glass jumps 3 -> 20 dB, and the lunch-room
# glass (ordinary) must become its OWN class (~3 dB). That un-fold makes v2 a
# 7-class model (cascade C). Ids below are the v2 (7-class) scheme.
V2 = dict(
    f_ref_mhz=2442.0,
    # 9-anchor frequency union: 5 scanner bands + 4 assumed indoor-Tx bands
    freqs_mhz=[619.0, 627.0, 1935.0, 2442.0, 2510.0, 2600.0,
               3500.0, 5500.0, 6125.0],
    # L_ref at f_ref (dB per crossing) and per-material frequency exponent
    # gamma in L_w(f) = L_ref * (f/f_ref)**gamma. gamma is a SEED; the real
    # values come from fitting the Gflex multi-band logs (MODEL_CARD_v2 2.1).
    walls={            # id: (name, L_ref_dB, gamma)
        1: ("drywall_partition",   3.0,  0.45),
        2: ("concrete_masonry",   15.0,  0.75),
        3: ("core_service_area",  22.0,  0.65),
        5: ("exterior_glass_lowE", 20.0, 0.35),   # <-- 20 dB, measured facade
        6: ("interior_glass",      3.0,  0.35),   # <-- new class, normal glass
    },
    clutter={4: ("furniture", 0.30, 0.55)},       # dB/m at f_ref, exponent
    # enhancements, each a MODEL_CARD_v2 section; None = engine not written yet
    angle_of_incidence=None,   # §2.2 scale loss by sec(theta) about wall normal
    thickness_scaling=None,    # §2.3 Beer-Lambert per rasterized run length
    diffraction=None,          # §2.4 knife-edge/UTD -> then DELETE saturation
    faf_db_per_floor=None,     # §2.5 floor attenuation, for vertical/outdoor
    o2i_lowE_db=25.0,          # 20 dB low-E facade shifts O2I ~15 -> ~25 dB
)

# Straight-ray multiwall is only trusted for a handful of walls; beyond that,
# measured indoor loss grows sublinearly because energy arrives by diffraction
# and corridor paths (the spec's 12.3-2 ladder rung). Until that rung is
# implemented, total obstruction loss is linear up to OBS_LINEAR_DB and then
# saturates smoothly, capping at OBS_LINEAR_DB + OBS_SAT_EXTRA_DB.
OBS_LINEAR_DB = 40.0
OBS_SAT_EXTRA_DB = 50.0


def saturate_obstruction(x):
    x = np.asarray(x, np.float32)
    over = x - OBS_LINEAR_DB
    return np.where(x <= OBS_LINEAR_DB, x,
                    OBS_LINEAR_DB + OBS_SAT_EXTRA_DB * -np.expm1(-np.maximum(over, 0) / OBS_SAT_EXTRA_DB))

# A.3: 3GPP TR 38.901 O2I penetration at ~3.5 GHz
O2I_DB = dict(low_loss=15.0, high_loss=28.0)


# --------------------------------------------------------------------------
# A.1 grid preparation
# --------------------------------------------------------------------------
def prepare_grid():
    src = np.load(ROOT / "STEP_2" / "material_grid_consolidated.npy")
    outside = np.load(ROOT / "STEP_2" / "outside_mask.npy")
    meta1 = json.loads((ROOT / "STEP_1" / "floorplan_meta.json").read_text())
    mpp0 = meta1["meters_per_px"]

    fold = np.zeros(256, np.uint8)
    for k, v in FOLD_8_TO_6.items():
        fold[k] = v
    src6 = fold[src]

    Hs, Ws = src6.shape
    width_m = Ws * mpp0
    cell = width_m / W_MODEL                       # square cells
    rows = int(round(Hs * mpp0 / cell))            # floor height in cells
    pad_top = (H_MODEL - rows) // 2

    loss = np.array([m["loss_db"] for m in MATERIALS6], np.float32)
    grid = np.zeros((H_MODEL, W_MODEL), np.uint8)  # padding = air
    inside = np.zeros((H_MODEL, W_MODEL), bool)
    # Pooling: walls (loss_db > 0) by ANY-hit with max loss, so 2-px walls
    # survive the 2.57x stride; furniture by MAJORITY vote vs air, so clutter
    # coverage isn't inflated by single-pixel chairs.
    ye = np.linspace(0, Hs, rows + 1)
    xe = np.linspace(0, Ws, W_MODEL + 1)
    for i in range(rows):
        y0, y1 = int(ye[i]), max(int(ye[i + 1]), int(ye[i]) + 1)
        for j in range(W_MODEL):
            x0, x1 = int(xe[j]), max(int(xe[j + 1]), int(xe[j]) + 1)
            blk = src6[y0:y1, x0:x1].ravel()
            wall_losses = loss[blk]
            if wall_losses.max() > 0:
                grid[pad_top + i, j] = blk[np.argmax(wall_losses)]
            elif (blk == 4).mean() >= 0.5:
                grid[pad_top + i, j] = 4
            inside[pad_top + i, j] = (~outside[y0:y1, x0:x1]).mean() >= 0.5

    # Core cleanup: real service cores (elevator banks, stairs, WC) are
    # several m^2; small class-3 fragments are rasterizer junction misfires
    # that would each charge 20 dB per crossing. Keep components >= 40 cells
    # (~1.2 m^2), downgrade the rest to drywall.
    from scipy import ndimage
    lbl, n = ndimage.label(grid == 3)
    if n:
        areas = ndimage.sum_labels(np.ones_like(lbl), lbl, np.arange(1, n + 1))
        small = np.isin(lbl, np.nonzero(areas < 40)[0] + 1)
        grid[small] = 1

    walkable = np.isin(grid, [0, 4]) & inside      # Rule R8
    np.save(SIM / "grid_model.npy", grid)
    np.save(SIM / "inside_mask.npy", inside)
    np.save(SIM / "walkable_mask.npy", walkable)
    return grid, inside, walkable, cell, mpp0


# --------------------------------------------------------------------------
# A.2 indoor multi-wall model
# --------------------------------------------------------------------------
def fspl_1m_db(f_mhz):
    return 32.44 + 20 * np.log10(f_mhz) - 60.0


def loss_table(f_mhz, jitter=None):
    """(per-crossing dB, per-meter dB) for the 6 classes at frequency f,
    optionally multiplied by a per-class jitter vector (B.2)."""
    if PHYSICS_VARIANT == "v1.2":
        L = np.zeros(6, np.float32)
        A = np.zeros(6, np.float32)
        for i, tab in LOSS_DB_V2.items():
            L[i] = tab[int(f_mhz)]
        for i, tab in LOSS_PER_M_V2.items():
            A[i] = tab[int(f_mhz)]
    else:
        mult = FREQ_LOSS_MULT[float(f_mhz)]
        L = np.array([m["loss_db"] for m in MATERIALS6], np.float32) * mult
        A = np.array([m["loss_per_m_db"] for m in MATERIALS6], np.float32) * mult
    if jitter is not None:
        j = np.asarray(jitter, np.float32)
        L, A = L * j, A * j
    L[0] = A[0] = 0.0
    return L, A


def pathloss_map(grid, tx_xy, f_mhz, cell_size_m, jitter=None,
                 return_crossings=False, chunk_rows=16):
    """PL (dB, float32 HxW) from tx_xy = (x, y) in cell coords.
    Rule R3: k_i counts contiguous runs of class i along the ray."""
    H, W = grid.shape
    L, A = loss_table(f_mhz, jitter)
    tx_x, tx_y = float(tx_xy[0]), float(tx_xy[1])

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    dist_c = np.hypot(gx - tx_x, gy - tx_y)
    d_m = np.maximum(dist_c * cell_size_m, D0_M)          # clamp at d0
    pl = (fspl_1m_db(f_mhz) + 10 * N_EXP * np.log10(d_m)).astype(np.float32)

    K = int(np.ceil(dist_c.max() / STEP_CELL)) + 1
    t = np.linspace(0.0, 1.0, K, dtype=np.float32)
    k_tot = np.zeros((H, W), np.int16) if return_crossings else None

    for r0 in range(0, H, chunk_rows):
        r1 = min(r0 + chunk_rows, H)
        ex = gx[r0:r1].ravel()[:, None]
        ey = gy[r0:r1].ravel()[:, None]
        xi = np.clip((tx_x + t * (ex - tx_x)).round(), 0, W - 1).astype(np.int32)
        yi = np.clip((tx_y + t * (ey - tx_y)).round(), 0, H - 1).astype(np.int32)
        mats = grid[yi, xi]
        wall = np.zeros(mats.shape[0], np.float32)
        ktot = np.zeros(mats.shape[0], np.int32)
        for m in range(1, 6):
            if L[m] > 0:                             # per-crossing (R3)
                hit = mats == m
                runs = hit[:, 0].astype(np.int32) + \
                    (hit[:, 1:] & ~hit[:, :-1]).sum(axis=1, dtype=np.int32)
                wall += L[m] * runs
                ktot += runs
            if A[m] > 0:                             # per-meter (12.3-1 form)
                spacing_m = dist_c[r0:r1].ravel() / (K - 1) * cell_size_m
                wall += A[m] * (mats == m).sum(axis=1) * spacing_m
        pl[r0:r1] += saturate_obstruction(wall).reshape(r1 - r0, W)
        if return_crossings:
            k_tot[r0:r1] = ktot.reshape(r1 - r0, W).astype(np.int16)
    return (pl, k_tot) if return_crossings else pl


def arrival_time_indoor(grid, tx_xy, cell_size_m):
    """F.5 v1: T(p) = d(t, p) / c, Euclidean (seconds)."""
    H, W = grid.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    return np.hypot(gx - tx_xy[0], gy - tx_xy[1]) * cell_size_m / C_MPS


# --------------------------------------------------------------------------
# A.3 outdoor base-station plane-wave mode
# --------------------------------------------------------------------------
def facade_sources(grid, inside, bearing_deg):
    """Illuminated exterior-envelope cells for a wave arriving FROM compass
    bearing theta (0 = north = -y, 90 = east = +x). Outward normal is
    estimated from adjacent outside cells."""
    H, W = grid.shape
    th = np.radians(bearing_deg)
    u_from = np.array([np.sin(th), -np.cos(th)])   # building -> BS direction
    out = ~inside
    ys, xs = np.nonzero((grid == 5))
    # outward normal from all outside cells within Chebyshev radius 2,
    # weighted by 1/distance -- robust to the ragged pooled boundary
    offs = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)
            if (dy, dx) != (0, 0)]
    srcs = []
    for y, x in zip(ys, xs):
        n = np.zeros(2)
        for dy, dx in offs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and out[yy, xx]:
                w = 1.0 / np.hypot(dx, dy)
                n += (dx * w, dy * w)
        if np.linalg.norm(n) < 0.5:
            continue                                # interior glass, skip
        n = n / np.linalg.norm(n)
        if n @ u_from > 0.15:                       # faces the BS
            srcs.append((x, y, float(x * u_from[0] + y * u_from[1])))
    return srcs, u_from


def bs_maps(grid, inside, bearing_deg, f_mhz, cell_size_m,
            o2i_db=O2I_DB["low_loss"], src_stride=3):
    """Returns (gain_db, t_arrival_s). gain_db is path gain RELATIVE to the
    facade reference power P_ref (calibrated in Phase D):
        P_rx(p) = P_ref + gain_db(p).
    Each illuminated facade cell radiates P_ref - O2I inward through the
    multi-wall model; contributions combine in linear power (R7), averaged
    over sources so the result is P_ref-referenced, not source-count-scaled."""
    srcs, u_from = facade_sources(grid, inside, bearing_deg)
    srcs = srcs[::src_stride]
    if not srcs:
        raise ValueError(f"no illuminated facade at bearing {bearing_deg}")
    H, W = grid.shape
    lin = np.zeros((H, W), np.float64)
    proj = np.array([s[2] for s in srcs])
    t0 = (proj.max() - proj) * cell_size_m / C_MPS  # nearest-to-BS fires first

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    t_arr = np.full((H, W), np.inf, np.float32)
    for (x, y, _), t0s in zip(srcs, t0):
        pl = pathloss_map(grid, (x, y), f_mhz, cell_size_m)
        lin += 10.0 ** ((-o2i_db - pl) / 10.0)
        d = np.hypot(gx - x, gy - y) * cell_size_m
        t_arr = np.minimum(t_arr, t0s + d / C_MPS)
    gain = 10.0 * np.log10(lin / len(srcs))
    return gain.astype(np.float32), t_arr


# --------------------------------------------------------------------------
# manifest (R6 — single source of truth)
# --------------------------------------------------------------------------
def write_manifest(grid, walkable, cell_size_m, mpp0):
    manifest = dict(
        version="sim-v1.1",
        grid_shape=[H_MODEL, W_MODEL],
        cell_size_m=round(cell_size_m, 6),
        floor_rows=[int(np.nonzero(np.load(SIM / 'inside_mask.npy').any(1))[0][0]),
                    int(np.nonzero(np.load(SIM / 'inside_mask.npy').any(1))[0][-1] + 1)],
        scale_status=f"measured from QGIS GCPs: {mpp0:.4f} m/px source raster; "
                     "supersedes the 12.3 px/m estimate in the build doc",
        # 9th channel deviates from the doc's 8-channel table: without an
        # explicit distance input the net must infer 20log10(d) from a
        # sigma=2 blob and stalls ~16 dB val RMSE. Distance is geometry, not
        # a Tx parameter, so R2 (no power/gain inputs) is untouched.
        channels=["onehot_air", "onehot_drywall", "onehot_concrete",
                  "onehot_core", "onehot_furniture", "onehot_exterior",
                  "tx_gaussian_sigma2", "freq_feature", "log10_distance_m_over3"],
        tx_blob_sigma_cells=2.0,
        dist_channel_norm=3.0,
        # clip window widened from the doc's [40,150]: this floor's multiwall
        # + saturation tops out ~170 dB, and [40,150] would clip ~50% of cells
        norm=dict(pl_min_db=40.0, pl_range_db=130.0,
                  freq_log_lo_mhz=2400.0, freq_log_hi_mhz=6125.0),
        physics=dict(model="multiwall_motley_keenan_v1", n_exp=N_EXP,
                     d0_m=D0_M, fspl_1m_at_3500=round(fspl_1m_db(3500), 2),
                     ray_step_cells=STEP_CELL,
                     crossing_rule="contiguous runs count once (R3)",
                     obstruction_linear_db=OBS_LINEAR_DB,
                     obstruction_sat_extra_db=OBS_SAT_EXTRA_DB,
                     obstruction_note="linear to 40 dB then smooth saturation "
                     "capped at 90 dB: stand-in for the 12.3-2 diffraction "
                     "fill; remove when that ladder rung is implemented"),
        materials=MATERIALS6,
        material_fold_8to6={str(k): v for k, v in FOLD_8_TO_6.items()},
        freqs_mhz=FREQS_MHZ,
        freq_loss_mult={str(int(k)): v for k, v in FREQ_LOSS_MULT.items()},
        o2i_db=O2I_DB,
        speed_of_light_mps=C_MPS,
        ui=dict(bands=[dict(label="2.4 GHz Wi-Fi", f_mhz=2442),
                       dict(label="3.5 GHz NR n78", f_mhz=3500),
                       dict(label="5 GHz Wi-Fi", f_mhz=5500),
                       dict(label="6 GHz", f_mhz=6125)],
                tx_power_dbm=[0, 30], antenna_gain_dbi=[-2, 9],
                rsrp_good_dbm=-85,
                sigma_sf_db=dict(los=3.0, nlos=8.03),
                reliability_z=dict(**{"80": 0.84, "90": 1.28, "95": 1.65}),
                timelapse_slowdown=1e-8,
                # demo presets: typical hardware values; choosing one is an
                # explicit user action, so R10 (no silent defaults) holds
                presets=[
                    dict(label="Home Wi-Fi router (2.4 GHz)",
                         f_mhz=2442, tx_power_dbm=20, gain_dbi=2),
                    dict(label="Enterprise ceiling AP (5 GHz)",
                         f_mhz=5500, tx_power_dbm=17, gain_dbi=4),
                    dict(label="Wi-Fi 6E AP (6 GHz)",
                         f_mhz=6125, tx_power_dbm=18, gain_dbi=3),
                    dict(label="5G small cell n78 (3.5 GHz)",
                         f_mhz=3500, tx_power_dbm=24, gain_dbi=5),
                    dict(label="Phone hotspot (2.4 GHz)",
                         f_mhz=2442, tx_power_dbm=15, gain_dbi=0),
                ],
                # P_ref anchored to the walk test: the value that makes the
                # simulated BS map's indoor median equal the measured median
                # (-111.3 dBm over 843 on-floor NR points). Bearing unknown
                # until Phase D; 135 deg is a placeholder for demos.
                bs_preset=dict(label="Outdoor macro (walk-test calibrated)",
                               p_ref_dbm=11.0, bearing_deg=135,
                               f_mhz=3500,
                               note="level anchored to measured indoor median; "
                                    "bearing is a demo placeholder")),
        walkable_mask_sha256=hashlib.sha256(walkable.tobytes()).hexdigest()[:16],
        grid_sha256=hashlib.sha256(grid.tobytes()).hexdigest()[:16],
    )
    (SIM / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------
# A.4 unit tests
# --------------------------------------------------------------------------
def run_tests():
    cell = 0.2
    g = np.zeros((64, 64), np.uint8)
    g[:, 30:33] = 1                                  # 3-cell drywall wall
    pl, k = pathloss_map(g, (10, 32), 3500.0, cell, return_crossings=True)
    assert k[32, 50] == 1, f"3-cell wall must be ONE crossing, got {k[32, 50]}"
    L, A = loss_table(3500.0)
    d = 40 * cell
    expect = fspl_1m_db(3500) + 20 * np.log10(d) + L[1]
    got = pl[32, 50]
    assert abs(got - expect) < 0.3, f"PL {got:.2f} != {expect:.2f}"
    g[:, 40:41] = 2                                  # add a concrete wall
    pl2, k2 = pathloss_map(g, (10, 32), 3500.0, cell, return_crossings=True)
    assert k2[32, 50] == 2, "two separated walls = two crossings"
    assert abs((pl2 - pl)[32, 50] - L[2]) < 0.3, "second wall adds L_concrete"
    # frequency scaling: same geometry, higher f => higher loss everywhere
    pl6 = pathloss_map(g, (10, 32), 6125.0, cell)
    assert (pl6 - pl2)[32, 50] > 0, "6 GHz must lose more than 3.5 GHz"
    # LOS cell before the wall has zero crossings
    assert k2[32, 20] == 0, "cell before the wall must be LOS"
    # furniture: per-meter Beer-Lambert, run count irrelevant
    gf = np.zeros((64, 64), np.uint8)
    gf[:, 20:45] = 4                                 # 25 cells of clutter
    plf = pathloss_map(gf, (10, 32), 3500.0, cell)
    extra = plf[32, 50] - (fspl_1m_db(3500) + 20 * np.log10(d))
    expect_f = A[4] * 25 * cell
    assert abs(extra - expect_f) < 0.35, f"furniture {extra:.2f} != {expect_f:.2f}"
    print("A.4 unit tests PASSED")


def sample_maps():
    grid = np.load(SIM / "grid_model.npy")
    inside = np.load(SIM / "inside_mask.npy")
    manifest = json.loads((SIM / "manifest.json").read_text())
    cell = manifest["cell_size_m"]
    txs = json.loads((ROOT / "STEP_1" / "transmitters.json").read_text())["transmitters"]
    # STEP_1 pin -> model cell coords (source px -> meters -> cells, plus pad)
    mpp0 = 0.0679
    pad_top = manifest["floor_rows"][0]
    tx = (txs[0]["x_px"] * mpp0 / cell, txs[0]["y_px"] * mpp0 / cell + pad_top)

    t0 = time.time()
    pl = pathloss_map(grid, tx, 3500.0, cell)
    t_indoor = time.time() - t0
    t0 = time.time()
    gain, t_arr = bs_maps(grid, inside, 90.0, 3500.0, cell)
    t_bs = time.time() - t0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (SIM / "preview").mkdir(exist_ok=True)
    for arr, name, ttl in [(np.where(inside, pl, np.nan), "indoor_pl",
                            f"indoor PL @3.5GHz, Tx=green pin ({t_indoor:.2f}s)"),
                           (np.where(inside, gain, np.nan), "bs_gain",
                            f"BS gain re P_ref, bearing 90E ({t_bs:.1f}s)"),
                           (np.where(inside, t_arr * 1e9, np.nan), "bs_arrival",
                            "BS wavefront arrival (ns)")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(arr, cmap="turbo")
        walls = np.isin(grid, [1, 2, 3, 5])
        ov = np.zeros((*grid.shape, 4)); ov[walls] = (0, 0, 0, 0.7)
        ax.imshow(ov)
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout(); fig.savefig(SIM / "preview" / f"{name}.png", dpi=110)
        plt.close(fig)
    print(f"indoor map {t_indoor:.2f}s (target <=2s) | BS map {t_bs:.1f}s")
    print(f"indoor PL range {np.nanmin(pl):.1f}..{np.nanpercentile(np.where(inside, pl, np.nan), 99):.1f} dB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    if a.test:
        run_tests()
    if a.prepare:
        grid, inside, walkable, cell, mpp0 = prepare_grid()
        m = write_manifest(grid, walkable, cell, mpp0)
        print(f"grid {grid.shape}, cell {cell:.4f} m, "
              f"walkable {walkable.sum()} cells, floor rows {m['floor_rows']}")
    if a.sample:
        sample_maps()
