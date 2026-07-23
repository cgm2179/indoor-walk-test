#!/usr/bin/env python3
"""
engine_v2.py — raster geometry for the enhanced Motley-Keenan model.

`physics_v2.py` answers "how many dB does this crossing cost?".  This module
answers "what does the ray actually cross, at what angle, through how much
material?" and combines the direct and diffracted paths.

The three ideas that make v2 affordable:

 1. GEOMETRY IS FREQUENCY-INDEPENDENT.  A ray crosses the same walls at the
    same angles whatever the carrier.  So we trace once per source and then
    evaluate all nine bands with a table lookup.  v2 with 9 bands costs about
    what v1 cost for 1, which is the only reason a 9-band dataset is
    tractable at all.

 2. CROSSINGS ARE RUN BOUNDARIES.  A single np.nonzero on "the material
    changed" yields every crossing of every class in one pass, with its start
    index, end index and therefore its run length.  No per-material loop.

 3. DIFFRACTION IS A RELAY SOURCE.  Each selected building corner re-radiates
    inward exactly the way the existing outdoor-BS mode already treats facade
    cells, so the same traced-map machinery serves both.

usage:
  python SIM/engine_v2.py --test        # geometry self-checks on synthetic grids
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage

import physics_v2 as P

C_MPS = P.C_MPS
D0_M = 1.0
N_EXP = 2.0                       # Motley-Keenan convention: free space, the
                                  # walls ARE the environment (v1 bug #3)
STEP_CELL = 0.5                   # ray sampling step, in cells
K_QUANT = 64                      # sample counts are rounded up to a multiple
                                  # of this; any deterministic function of the
                                  # ray length preserves reciprocity, and 64
                                  # wastes far less than rounding to powers of 2
K_QUANTUM = 64                    # sample counts are rounded up to a multiple
                                  # of this; any deterministic function of the
                                  # ray length preserves reciprocity, and a
                                  # fixed quantum wastes far fewer samples than
                                  # rounding to powers of two
K_QUANTUM = 64                    # sample counts are rounded up to a multiple
                                  # of this; any deterministic function of the
                                  # ray length preserves reciprocity, and a
                                  # fixed quantum wastes far fewer samples than
                                  # rounding to powers of two


# ---------------------------------------------------------------------------
# grid-derived geometry, computed once per grid
# ---------------------------------------------------------------------------
def wall_normals(grid, sigma=1.2):
    """Unit outward normal of the structure boundary at every cell.

    Estimated from the gradient of a Gaussian-blurred occupancy mask.  We only
    ever read this at run-START samples, which are boundary cells by
    construction, so the gradient is well conditioned wherever it is used.

    Returns (nx, ny) float32 arrays.  Deep inside a solid the gradient
    vanishes; those cells get (0,0) and the caller falls back to normal
    incidence, which is the conservative choice.
    """
    occ = (grid > 0).astype(np.float32)
    sm = ndimage.gaussian_filter(occ, sigma, mode="nearest")
    gy, gx = np.gradient(sm)
    mag = np.hypot(gx, gy)
    ok = mag > 1e-4
    nx = np.where(ok, -gx / np.maximum(mag, 1e-9), 0.0).astype(np.float32)
    ny = np.where(ok, -gy / np.maximum(mag, 1e-9), 0.0).astype(np.float32)
    return nx, ny


def measure_nominal_widths(grid, n_classes, cell_size_m):
    """Median rasterised width, in metres, of each material class.

    This is the calibration that connects raster run length to real thickness.
    A CAD-drawn partition is ~2 source pixels wide whatever it is really made
    of, so the raster's run length is a RELATIVE measure: a run twice the
    class median means a wall twice as thick as that class's nominal
    construction thickness.  Measuring the median from the grid itself means
    the mapping survives a re-rasterisation at a different DPI, which a
    hard-coded pixel count would not.

    Scans rows and columns for contiguous runs and takes the median per class.
    """
    out = np.zeros(n_classes, np.float32)
    for c in range(1, n_classes):
        runs = []
        for arr in (grid, grid.T):
            hit = (arr == c)
            if not hit.any():
                continue
            pad = np.zeros((hit.shape[0], 1), bool)
            h = np.hstack([pad, hit, pad])
            d = np.diff(h.astype(np.int8), axis=1)
            starts = np.nonzero(d == 1)
            ends = np.nonzero(d == -1)
            if len(starts[1]):
                runs.append(ends[1] - starts[1])
        if runs:
            out[c] = float(np.median(np.concatenate(runs))) * cell_size_m
        else:
            out[c] = cell_size_m
    return out


def find_diffracting_edges(grid, inside, radius=3, n_angles=72,
                           n_min=1.15, min_sep=6):
    """Locate convex structure corners that can diffract, and measure the
    wedge parameter n at each.

    Method: for every boundary structure cell, sample the occupancy on a
    circle of `radius` cells and find the largest contiguous AIR arc.  A flat
    wall face subtends exactly pi of air; a 90 degree external corner subtends
    3*pi/2.  The wedge parameter is then n = (air arc)/pi directly, which is
    the same n the Kouyoumjian-Pathak coefficient wants -- so the geometry
    measurement and the physics parameter are the same number, rather than the
    usual practice of assuming n=1.5 everywhere.

    Returns a list of dicts: x, y, n, face0 (radians, the 0-face direction).
    Candidates are thinned so no two are within `min_sep` cells.
    """
    occ = grid > 0
    bnd = occ & ~ndimage.binary_erosion(occ, np.ones((3, 3), bool))
    ys, xs = np.nonzero(bnd & _dilate(inside, 2))
    if len(ys) == 0:
        return []

    ang = np.arange(n_angles) * (2 * np.pi / n_angles)
    dx = np.round(radius * np.cos(ang)).astype(int)
    dy = np.round(radius * np.sin(ang)).astype(int)
    H, W = grid.shape

    cands = []
    for y, x in zip(ys, xs):
        yy = np.clip(y + dy, 0, H - 1)
        xx = np.clip(x + dx, 0, W - 1)
        air = ~occ[yy, xx]
        if air.all() or not air.any():
            continue
        start, length = _longest_circular_run(air)
        n_wedge = length * (2 * np.pi / n_angles) / np.pi
        if n_wedge < n_min or n_wedge > 2.0:
            continue
        # The UTD convention puts the exterior (air) region at phi in [0,n*pi]
        # measured from the 0-face, so the 0-face is where the air arc BEGINS
        # and phi increases THROUGH the air.  Taking the other end gives the
        # n-face, which pushes every phi' outside the valid range and switches
        # diffraction off entirely.
        face0 = ang[start % n_angles]
        # The corner cell is itself STRUCTURE, so tracing either diffraction
        # leg from it charges the ray for the very wall it is bending around
        # -- about 15 dB on each leg for concrete, which silently reduces the
        # diffracted field to nothing.  Anchor the relay a couple of cells out
        # along the bisector of the air arc instead.  The offset is ~0.35 m
        # against leg lengths of tens of metres, so the UTD geometry is still
        # referenced to the true edge; only the obstruction trace moves.
        bis = face0 + 0.5 * n_wedge * np.pi
        ax = ay = None
        for off in (2, 3, 4, 5):
            cx = int(round(x + off * np.cos(bis)))
            cy = int(round(y + off * np.sin(bis)))
            if 0 <= cx < W and 0 <= cy < H and not occ[cy, cx]:
                ax, ay = cx, cy
                break
        if ax is None:
            continue
        cands.append(dict(x=int(x), y=int(y), n=float(min(n_wedge, 2.0)),
                          face0=float(face0), ax=int(ax), ay=int(ay)))

    # thin: keep the sharpest corner (largest n) in each min_sep neighbourhood
    cands.sort(key=lambda c: -c["n"])
    kept = []
    for c in cands:
        if all((c["x"] - k["x"]) ** 2 + (c["y"] - k["y"]) ** 2 >= min_sep ** 2
               for k in kept):
            kept.append(c)
    return kept


def _dilate(mask, r):
    return ndimage.binary_dilation(mask, np.ones((2 * r + 1, 2 * r + 1), bool))


def _longest_circular_run(b):
    """Longest contiguous run of True in a circular boolean array.
    Returns (start_index, length)."""
    n = len(b)
    if b.all():
        return 0, n
    d = np.concatenate([b, b])
    best_len = best_start = 0
    cur = 0
    for i in range(2 * n):
        if d[i]:
            cur += 1
            if cur > best_len:
                best_len = cur
                best_start = i - cur + 1
        else:
            cur = 0
    return best_start % n, min(best_len, n)


# ---------------------------------------------------------------------------
# the ray tracer
# ---------------------------------------------------------------------------
def fspl_1m_db(f_mhz):
    """Free-space loss at 1 m.  32.44 + 20log10(f_MHz) - 60."""
    return 32.44 + 20.0 * np.log10(np.asarray(f_mhz, float)) - 60.0


class SceneV2:
    """Everything about a grid that does not depend on the source position.

    Built once and reused across all 2,500 Tx positions in Phase B.
    """

    def __init__(self, grid, inside, cell_size_m, lut=None, materials=None,
                 freqs_mhz=None, n_edges=8, edge_radius=3, n_relay_cache=24,
                 precompute=True, relay_dtype=np.float16):
        self.grid = np.ascontiguousarray(grid.astype(np.uint8))
        self.inside = inside
        self.cell = float(cell_size_m)
        self.materials = materials if materials is not None else P.MATERIALS7
        self.freqs = np.asarray(
            freqs_mhz if freqs_mhz is not None else P.FREQS_MHZ_V2, float)
        self.lut = lut if lut is not None else P.CrossingLUT(
            self.materials, self.freqs)
        self.n_classes = len(self.materials)
        self.nx, self.ny = wall_normals(self.grid)
        self.w_ref = measure_nominal_widths(self.grid, self.n_classes, self.cell)
        self.t_ref = np.array([m.get("t_ref_m", 0.0)
                               for m in self.materials], np.float32)
        self.per_metre = np.array([bool(m.get("per_metre"))
                                   for m in self.materials])
        self.edges = find_diffracting_edges(self.grid, inside,
                                            radius=edge_radius)
        self.n_edges = n_edges
        H, W = self.grid.shape
        self.gx, self.gy = np.meshgrid(np.arange(W, dtype=np.float32),
                                       np.arange(H, dtype=np.float32))
        self.fspl1 = fspl_1m_db(self.freqs)

        # ---- relay cache --------------------------------------------------
        # The obstruction map radiating FROM a corner does not depend on where
        # the transmitter is.  Recomputing it per transmitter made a 9-band
        # map cost 26 s and put Phase B at ~18 hours; precomputing a spread of
        # corners once per grid makes the per-transmitter diffraction cost
        # essentially just the UTD arithmetic.  float16 keeps 24 relays at
        # ~50 MB for a 256x448 grid, and obstruction is a smooth dB quantity
        # where half precision is worth ~0.01 dB.
        self.relays = self._pick_relays(n_relay_cache) if precompute else []
        self.relay_obs = None
        if self.relays:
            self.relay_obs = np.empty(
                (len(self.relays), len(self.freqs), H, W), relay_dtype)
            for i, e in enumerate(self.relays):
                self.relay_obs[i] = self.obstruction_maps(
                    (float(e["ax"]), float(e["ay"]))).astype(relay_dtype)

    def _pick_relays(self, k):
        """Farthest-point sample of the detected corners.

        Ranking corners purely by sharpness clusters every pick around one
        busy junction; spreading them means that wherever the transmitter
        lands, some cached corner is usefully placed relative to it.
        """
        if not self.edges or k <= 0:
            return []
        pts = np.array([[e["ax"], e["ay"]] for e in self.edges], float)
        sharp = np.array([e["n"] for e in self.edges])
        chosen = [int(np.argmax(sharp))]
        d2 = ((pts - pts[chosen[0]]) ** 2).sum(1)
        while len(chosen) < min(k, len(self.edges)):
            nxt = int(np.argmax(d2))
            if d2[nxt] <= 0:
                break
            chosen.append(nxt)
            d2 = np.minimum(d2, ((pts - pts[nxt]) ** 2).sum(1))
        return [self.edges[i] for i in chosen]

    # -- core trace ---------------------------------------------------------
    def obstruction_maps(self, src_xy):
        """Total obstruction loss (dB, excluding free space) from a source at
        src_xy, for every frequency: array (n_freq, H, W).

        One geometric pass, all bands.

        Cells are processed in equal-count DISTANCE bands rather than row
        slices.  The sample count K along a ray is set by that ray's length,
        so a row slice -- which always spans the full width and therefore
        nearly the full distance range -- forces every cell to be sampled as
        finely as the farthest corner.  Banding by distance cuts the sampling
        work roughly in half and keeps the per-band memory flat.
        """
        grid = self.grid
        H, W = grid.shape
        nf = len(self.freqs)
        sx, sy = float(src_xy[0]), float(src_xy[1])

        dist_flat = np.hypot(self.gx.ravel() - sx, self.gy.ravel() - sy)
        out = np.zeros((nf, H * W), np.float32)

        # Group cells by the sample count THEIR OWN ray needs, quantised to
        # powers of two.  Two things fall out of this:
        #   * short rays are not sampled as finely as the farthest corner,
        #     which is most of the speed-up over uniform chunking;
        #   * K depends only on the ray's length, and d(A,B) == d(B,A), so
        #     both directions get identical sampling.  Keying K to a batch's
        #     max distance instead made path loss non-reciprocal by up to
        #     12 dB, because the two directions landed in different batches.
        need = np.ceil(dist_flat / STEP_CELL).astype(np.int64) + 2
        kq = ((np.maximum(need, K_QUANTUM) + K_QUANTUM - 1)
              // K_QUANTUM) * K_QUANTUM
        exf = self.gx.ravel()
        eyf = self.gy.ravel()
        for K in np.unique(kq):
            sel = np.nonzero(kq == K)[0]
            if sel.size == 0:
                continue
            K = int(K)
            d_cell = dist_flat[sel]
            t = np.linspace(0.0, 1.0, K, dtype=np.float32)
            ex = exf[sel]
            ey = eyf[sel]
            n_cell = ex.size
            xi = np.clip(np.rint(sx + t * (ex[:, None] - sx)),
                         0, W - 1).astype(np.int32)
            yi = np.clip(np.rint(sy + t * (ey[:, None] - sy)),
                         0, H - 1).astype(np.int32)
            mats = grid[yi, xi]

            # metres between consecutive samples, per receiver cell
            seg_m = (d_cell / (K - 1) * self.cell).astype(np.float32)

            # --- crossings = maximal runs of constant material -------------
            same = mats[:, 1:] == mats[:, :-1]
            starts = np.empty_like(mats, bool)
            ends = np.empty_like(mats, bool)
            starts[:, 0] = True
            starts[:, 1:] = ~same
            ends[:, -1] = True
            ends[:, :-1] = ~same
            si_r, si_c = np.nonzero(starts)
            ei_r, ei_c = np.nonzero(ends)
            # np.nonzero returns row-major order and every row has equal start
            # and end counts, so the two lists pair off elementwise.
            m_run = mats[si_r, si_c]
            keep = m_run > 0
            if not keep.any():
                continue
            si_r, si_c = si_r[keep], si_c[keep]
            ei_c = ei_c[keep]
            m_run = m_run[keep]
            run_cells = (ei_c - si_c + 1).astype(np.float32)
            run_len_m = run_cells * seg_m[si_r]

            # --- incidence angle, measured SYMMETRICALLY -------------------
            # Reading the normal only at the run START makes path loss
            # direction-dependent: start and end swap when the ray is
            # reversed, and on a 2-cell wall the blurred-occupancy gradient is
            # well conditioned on one face but not always the other, so one
            # direction gets the true angle and the other falls back to normal
            # incidence.  That showed up as a 25 dB reciprocity violation.
            # Averaging the valid faces is invariant under reversal, and it is
            # also a better estimate: two samples of the boundary instead of
            # one.
            rx = ex[si_r] - sx
            ry = ey[si_r] - sy
            rn = np.maximum(np.hypot(rx, ry), 1e-6)
            cos_acc = np.zeros(si_r.size, np.float32)
            n_ok = np.zeros(si_r.size, np.float32)
            for cc in (si_c, ei_c):
                wx = self.nx[yi[si_r, cc], xi[si_r, cc]]
                wy = self.ny[yi[si_r, cc], xi[si_r, cc]]
                ok = (wx * wx + wy * wy) > 0.25
                cos_acc += np.where(ok, np.abs((rx * wx + ry * wy) / rn), 0.0)
                n_ok += ok
            cos_th = np.where(n_ok > 0, cos_acc / np.maximum(n_ok, 1), 1.0)
            cos_th = np.clip(cos_th, 0.06, 1.0).astype(np.float32)

            # --- thickness -------------------------------------------------
            # run_len_m is measured ALONG the ray; the physical wall thickness
            # is that projected onto the normal.  Then scale by the class's
            # nominal construction thickness over its measured raster width,
            # so a run twice the class median reads as twice the thickness.
            t_raster = run_len_m * cos_th
            scale = self.t_ref[m_run] / np.maximum(self.w_ref[m_run], 1e-6)
            thick = (t_raster * scale).astype(np.float32)

            # --- bulk clutter accrues per metre, not per crossing ----------
            is_pm = self.per_metre[m_run]

            # --- accumulate, all frequencies in one gather -----------------
            per = self.lut.loss_all_bands(m_run, cos_th, thick)   # (nf, ncross)
            if is_pm.any():
                per = np.where(is_pm[None, :], per * run_len_m[None, :], per)
            for fi in range(nf):
                np.add.at(out[fi], sel, np.bincount(
                    si_r, weights=per[fi], minlength=n_cell))
        return out.reshape(nf, H, W)

    def direct_maps(self, tx_xy, obstruction=None):
        """Direct-path loss (dB), array (n_freq, H, W)."""
        if obstruction is None:
            obstruction = self.obstruction_maps(tx_xy)
        d_m = np.maximum(np.hypot(self.gx - tx_xy[0], self.gy - tx_xy[1])
                         * self.cell, D0_M)
        fs = (self.fspl1[:, None, None] +
              10.0 * N_EXP * np.log10(d_m)[None, :, :]).astype(np.float32)
        return fs + obstruction, fs

    # -- diffraction --------------------------------------------------------
    def select_edges(self, tx_xy, obs_direct_ref):
        """Pick the K cached relays most worth using for this transmitter.

        Ranked by how cheaply the Tx reaches the relay's air-side anchor: a
        corner the transmitter cannot see is a corner that cannot relay
        anything.  Sharper wedges get a small bonus.
        """
        pool = self.relays if self.relays else self.edges
        if not pool:
            return []
        ex = np.array([e["ax"] for e in pool])
        ey = np.array([e["ay"] for e in pool])
        cost = obs_direct_ref[ey, ex] - 2.0 * np.array([e["n"] for e in pool])
        order = np.argsort(cost)[:self.n_edges]
        return [(int(i), pool[int(i)]) for i in order]

    def diffracted_maps(self, tx_xy, obs_direct, edges=None, ref_band=None):
        """Sum of diffracted-path powers, (n_freq, H, W), in LINEAR power
        relative to the 1 m reference.

        For each relay corner the total diffracted path loss is

            PL = PL_utd(phi, phi', s, s', n, f)      [Block C, all spreading]
                 + obstruction(Tx -> corner anchor)  [read off the direct map]
                 + obstruction(anchor -> Rx)         [read off the relay cache]

        which is why the direct map is computed first: its value at the anchor
        cells IS the first leg, for free.
        """
        nf = len(self.freqs)
        H, W = self.grid.shape
        ref = nf // 2 if ref_band is None else ref_band
        if edges is None:
            edges = self.select_edges(tx_xy, obs_direct[ref])
        acc = np.zeros((nf, H, W), np.float64)
        if not edges:
            return acc

        txx, txy = float(tx_xy[0]), float(tx_xy[1])
        for idx, e in edges:
            exx, eyy = float(e["x"]), float(e["y"])
            s_p = max(np.hypot(txx - exx, txy - eyy) * self.cell, D0_M)
            s = np.maximum(np.hypot(self.gx - exx, self.gy - eyy) * self.cell,
                           D0_M)
            phi_p = (np.arctan2(txy - eyy, txx - exx) - e["face0"]) % (2 * np.pi)
            if phi_p > e["n"] * np.pi:
                continue                      # transmitter is behind the wedge
            phi = (np.arctan2(self.gy - eyy, self.gx - exx)
                   - e["face0"]) % (2 * np.pi)
            valid = phi <= e["n"] * np.pi
            if not valid.any():
                continue
            if self.relay_obs is not None:
                obs_leg2 = self.relay_obs[idx].astype(np.float32)
            else:
                obs_leg2 = self.obstruction_maps((float(e["ax"]),
                                                  float(e["ay"])))
            leg1 = obs_direct[:, e["ay"], e["ax"]]
            for fi, f in enumerate(self.freqs):
                pl = P.utd_pathloss_db(phi, phi_p, s, s_p, f, n=e["n"],
                                       soft=True)
                pl = pl + leg1[fi] + obs_leg2[fi]
                acc[fi] += np.where(valid, 10.0 ** (-pl / 10.0), 0.0)
        return acc

    # -- the public entry point --------------------------------------------
    def pathloss_maps(self, tx_xy, with_diffraction=True, edges=None):
        """Total path loss (dB) for every frequency: (n_freq, H, W).

        Direct and diffracted contributions combine in LINEAR POWER (rule R7),
        then the result is floored at free space -- a passive obstructed
        channel cannot deliver more than free space, and that floor is exactly
        the D.3 sanity check the project already runs.

        There is NO obstruction saturation here.  v1 needed one because a
        straight-ray-only model sends deep-shadow loss to +200 dB; with Block C
        supplying the energy that really arrives around corners, the loss
        self-limits for the right reason (DECISIONS D10: do not stack them).
        """
        obs = self.obstruction_maps(tx_xy)
        pl_direct, fs = self.direct_maps(tx_xy, obstruction=obs)
        lin = 10.0 ** (-pl_direct.astype(np.float64) / 10.0)
        if with_diffraction:
            lin = lin + self.diffracted_maps(tx_xy, obs, edges=edges)
        pl = (-10.0 * np.log10(np.maximum(lin, 1e-300))).astype(np.float32)
        return np.maximum(pl, fs)

    def arrival_time(self, tx_xy):
        """T(p) = d/c, seconds (F.5 v1 form, unchanged)."""
        return (np.hypot(self.gx - tx_xy[0], self.gy - tx_xy[1])
                * self.cell / C_MPS).astype(np.float32)


# ---------------------------------------------------------------------------
# synthetic scenes for testing (the real grid lives in SIM/grid_model.npy)
# ---------------------------------------------------------------------------
def synth_slab_scene(cell=0.1744, H=64, W=64, mat=2, x0=30, x1=33):
    """A single vertical wall: the analytic test case."""
    g = np.zeros((H, W), np.uint8)
    g[:, x0:x1] = mat
    inside = np.ones((H, W), bool)
    return g, inside, cell


def synth_floor_scene(cell=0.1744, H=256, W=448, seed=0):
    """A synthetic office floor with the same gross layout as the real 7th
    floor: glass envelope, central service core, drywall cellular offices
    around the perimeter, open-plan furniture bays.  Used so the engine can be
    exercised end to end without shipping the real grid.
    """
    rng = np.random.default_rng(seed)
    g = np.zeros((H, W), np.uint8)
    inside = np.zeros((H, W), bool)
    top, bot, lft, rgt = 32, 224, 4, W - 4
    inside[top:bot, lft:rgt] = True
    # exterior low-E glass envelope (class 5)
    g[top:top + 2, lft:rgt] = 5
    g[bot - 2:bot, lft:rgt] = 5
    g[top:bot, lft:lft + 2] = 5
    g[top:bot, rgt - 2:rgt] = 5
    # central service core (class 3), two lift banks
    for cx in (int(W * 0.42), int(W * 0.58)):
        g[110:150, cx - 14:cx + 14] = 3
    # concrete spine walls (class 2)
    g[104:108, 60:W - 60] = 2
    g[152:156, 60:W - 60] = 2
    # cellular offices along both long facades: drywall partitions (class 1)
    for x in range(20, W - 20, 26):
        g[top + 2:top + 34, x:x + 2] = 1
        g[bot - 34:bot - 2, x:x + 2] = 1
    g[top + 32:top + 34, 20:W - 20] = 1
    g[bot - 34:bot - 32, 20:W - 20] = 1
    # interior glass meeting room (class 6)
    g[112:148, 232:236] = 6
    g[112:116, 232:300] = 6
    g[144:148, 232:300] = 6
    # open-plan furniture bays (class 4)
    for y0 in (58, 172):
        for x0 in range(30, W - 40, 40):
            g[y0:y0 + 26, x0:x0 + 28] = np.where(
                rng.random((26, 28)) < 0.55, 4, g[y0:y0 + 26, x0:x0 + 28])
    # doorways: punch gaps so the floor is connected
    for x in range(20, W - 20, 26):
        g[top + 32:top + 34, x + 8:x + 18] = 0
        g[bot - 34:bot - 32, x + 8:x + 18] = 0
    g[104:108, 200:216] = 0
    g[152:156, 200:216] = 0
    g[~inside] = 0
    walk = np.isin(g, [0, 4]) & inside
    return g, inside, walk, cell


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def _test_crossing_extraction():
    """Run detection must count a thick wall once and separated walls twice --
    rule R3, carried over from v1 and re-verified on the new code path."""
    g, inside, cell = synth_slab_scene()
    sc = SceneV2(g, inside, cell, n_edges=0)
    obs = sc.obstruction_maps((10, 32))
    fi = int(np.argmin(np.abs(sc.freqs - 3500.0)))
    # one 3-cell wall
    one = obs[fi, 32, 50]
    # add a second, separated wall
    g2 = g.copy()
    g2[:, 40:41] = 2
    sc2 = SceneV2(g2, inside, cell, n_edges=0)
    two = sc2.obstruction_maps((10, 32))[fi, 32, 50]
    assert two > one, "a second wall must add loss"
    # a cell before any wall is unobstructed
    assert obs[fi, 32, 20] == 0.0, "LOS cell must have zero obstruction"
    # Thickness scaling has to be tested WITHIN one grid.  w_ref is the class
    # median measured from the grid, so a grid where every wall is 6 cells
    # simply redefines the nominal -- and should, since the raster width of a
    # CAD line carries no absolute thickness information.  What must hold is
    # that a wall thicker than its class median costs proportionally more.
    H2, W2 = 64, 96
    g4 = np.zeros((H2, W2), np.uint8)
    g4[0:32, 30:32] = 2                     # thin run, 2 cells
    g4[32:64, 30:34] = 2                    # thick run, 4 cells
    ins4 = np.ones((H2, W2), bool)
    sc4 = SceneV2(g4, ins4, cell, n_edges=0)
    o4 = sc4.obstruction_maps((10, 16))
    thin_l = o4[fi, 16, 60]
    o4b = sc4.obstruction_maps((10, 48))
    thick_l = o4b[fi, 48, 60]
    ratio = thick_l / max(thin_l, 1e-6)
    assert 1.4 < ratio < 2.6, \
        f"2x raster thickness gave {ratio:.2f}x loss ({thin_l:.1f} -> {thick_l:.1f} dB)"
    print(f"  [geom] crossings: 3-cell wall {one:.1f} dB, +2nd wall "
          f"{two:.1f} dB; 2x thickness -> {ratio:.2f}x loss "
          f"({thin_l:.1f} -> {thick_l:.1f} dB)")


def _test_angle_of_incidence():
    """Oblique crossings must cost more than normal ones -- but SUB-secant.

    The naive enhancement (L_eff = L_w / cos theta_i, the secant law in the
    v2 scaffold's todo-angle cell) is wrong for high-permittivity building
    materials.  Snell refracts the ray toward the normal INSIDE the wall: at
    56 degrees incidence on concrete (eps' = 5.24) the internal ray is at only
    21 degrees, so the absorbing path grows 7%, not 79%.  The extra loss at
    oblique incidence is mostly increased Fresnel REFLECTION, not a longer
    path, and the exact slab captures that automatically.

    So the physical bracket is:
        1/cos(theta_t)  <=  L(theta)/L(0)  <=  1/cos(theta_i)
    and a model sitting at the upper bound is over-attenuating oblique rays.
    """
    cell = 0.1744
    H, W = 321, 161
    g = np.zeros((H, W), np.uint8)
    g[:, 80:83] = 2                      # vertical wall through the middle
    inside = np.ones((H, W), bool)
    sc = SceneV2(g, inside, cell, n_edges=0, precompute=False)
    fi = int(np.argmin(np.abs(sc.freqs - 3500.0)))
    tx = (20, 160)
    obs = sc.obstruction_maps(tx)
    got = {}
    for dy in (0, 60, 110, 150):
        theta = np.degrees(np.arctan2(dy, 120 - 20))
        got[round(theta)] = float(obs[fi, 160 + dy, 120])
    ths = sorted(got)
    vals = [got[t] for t in ths]
    assert vals[0] < vals[1] < vals[2] < vals[3], f"loss must rise: {got}"

    eps_re = P.permittivity("concrete", 3500.0)[0].real
    for th_deg, v in zip(ths[1:], vals[1:]):
        th = np.radians(th_deg)
        lo = 1.0 / np.sqrt(1.0 - (np.sin(th) / np.sqrt(eps_re)) ** 2)
        hi = 1.0 / np.cos(th)
        ratio = v / vals[0]
        assert lo * 0.9 <= ratio <= hi * 1.05, \
            f"{th_deg} deg: ratio {ratio:.2f} outside [{lo:.2f}, {hi:.2f}]"
    th_max = np.radians(ths[-1])
    assert vals[-1] / vals[0] < 1.0 / np.cos(th_max) * 0.95, \
        "loss should be strictly SUB-secant; a secant law over-attenuates"
    print(f"  [geom] angle of incidence: {vals[0]:.1f} dB at 0 deg -> "
          f"{vals[-1]:.1f} dB at {ths[-1]} deg (ratio {vals[-1]/vals[0]:.2f}; "
          f"secant would say {1/np.cos(th_max):.2f}, Snell floor "
          f"{1/np.sqrt(1-(np.sin(th_max)/np.sqrt(eps_re))**2):.2f})")


def _test_frequency_consistency():
    """All bands come from one geometric trace; each must still be internally
    consistent and ordered."""
    g, inside, walk, cell = synth_floor_scene()
    sc = SceneV2(g, inside, cell, n_edges=0)
    obs = sc.obstruction_maps((100.0, 128.0))
    med = [float(np.median(obs[fi][inside])) for fi in range(len(sc.freqs))]
    # 619 and 627 MHz differ by 8 MHz; their median obstruction can tie to
    # within float noise, so allow a 0.05 dB slack on adjacent bands.
    assert all(b >= a - 0.05 for a, b in zip(med, med[1:])), \
        f"median obstruction must be ~monotone in frequency: {med}"
    assert med[-1] > med[0] + 20.0, "6 GHz must be much lossier than 619 MHz"
    print(f"  [geom] 9-band monotone: median obstruction "
          f"{med[0]:.1f} dB @619 MHz -> {med[-1]:.1f} dB @6125 MHz")


def _test_edges_and_wedges():
    g, inside, walk, cell = synth_floor_scene()
    edges = find_diffracting_edges(g, inside)
    assert len(edges) > 10, f"expected many corners, found {len(edges)}"
    ns = np.array([e["n"] for e in edges])
    assert np.all(ns >= 1.15) and np.all(ns <= 2.0)
    assert ns.mean() > 1.3, f"mean wedge parameter {ns.mean():.2f} looks flat"
    # a plain slab has no convex corners in the interior
    g2, ins2, _ = synth_slab_scene(H=64, W=64)
    e2 = find_diffracting_edges(g2, ins2)
    assert len(e2) <= 4, f"a straight wall should not spawn corners: {len(e2)}"
    print(f"  [geom] edge finder: {len(edges)} corners, "
          f"mean wedge n={ns.mean():.2f} (1.5 = 90 deg corner)")


def _test_diffraction_fills_shadow():
    """The point of Block C: shadow must be filled by energy that bends around
    a corner, not by an arbitrary saturation cap.

    Geometry is the canonical one -- a walled room off a corridor with a
    single doorway.  Cells straight through the door are lit directly; the
    room's far corners are reachable only by diffracting at the door jambs.
    An earlier version of this test used a thin free-standing wall, where the
    honest answer is that diffraction should NOT matter: punching through
    0.7 m of concrete costs 22 dB and detouring 20 m around the end costs
    47 dB, so the direct path rightly wins.
    """
    cell = 0.1744
    H = W = 200
    g = np.zeros((H, W), np.uint8)
    g[60:140, 100:106] = 3                       # left wall, high-loss class
    g[60:66, 100:180] = 3
    g[134:140, 100:180] = 3
    g[60:140, 174:180] = 3
    g[95:105, 100:106] = 0                       # doorway
    inside = np.ones((H, W), bool)
    sc = SceneV2(g, inside, cell, n_edges=6, n_relay_cache=12)
    fi = int(np.argmin(np.abs(sc.freqs - 3500.0)))
    tx = (40.0, 100.0)
    a = sc.pathloss_maps(tx, with_diffraction=False)[fi]
    b = sc.pathloss_maps(tx, with_diffraction=True)[fi]

    room = np.zeros((H, W), bool)
    room[68:132, 108:172] = True
    fill = float((a - b)[room].mean())
    assert fill > 3.0, f"diffraction only filled {fill:.1f} dB inside the room"
    corner = float(a[125, 165] - b[125, 165])
    assert corner > 8.0, f"far corner gained only {corner:.1f} dB"
    # straight through the doorway is already lit: diffraction must not touch it
    assert abs(a[100, 140] - b[100, 140]) < 0.5, "lit path was disturbed"
    # nor the open area on the transmitter's side
    near = ~room & (np.hypot(sc.gx - 40.0, sc.gy - 100.0) < 50)
    assert abs(float((a - b)[near].mean())) < 0.1, "LOS region polluted"
    # diffraction can only ADD power
    assert np.all(b <= a + 1e-3), "diffraction cannot increase loss"
    # energy conservation: never below free space
    d_m = np.maximum(np.hypot(sc.gx - 40.0, sc.gy - 100.0) * cell, 1.0)
    fs = fspl_1m_db(sc.freqs[fi]) + 20 * np.log10(d_m)
    assert np.all(b >= fs - 1e-3), "a cell fell below free-space loss"
    print(f"  [geom] diffraction: room mean fill {fill:.1f} dB, far corner "
          f"{corner:.1f} dB, lit path and LOS region untouched, never < FSPL")


def _test_no_saturation_needed():
    """v1 capped obstruction at 90 dB because straight rays ran away.  The
    claim v2 makes is not that path loss becomes small, but that the SATURATION
    CONSTANT IS NO LONGER NEEDED: the energy that fills deep shadow now comes
    from Block C, for a physical reason, and it materially improves how much
    of the map lands inside the trainable dynamic range.

    Honest residual: single-order diffraction from a finite set of corners
    under-fills the very deepest pockets.  What really reaches those is double
    diffraction and corridor waveguiding, neither of which v2 models -- so a
    few tenths of a percent of cells still sit above the clip ceiling.  That
    is a stated limitation, not something to paper over with a cap.
    """
    g, inside, walk, cell = synth_floor_scene()
    sc = SceneV2(g, inside, cell, n_edges=8, n_relay_cache=16)
    fi = int(np.argmin(np.abs(sc.freqs - 3500.0)))
    tx = (100.0, 128.0)
    off = sc.pathloss_maps(tx, with_diffraction=False)[fi][inside]
    on = sc.pathloss_maps(tx, with_diffraction=True)[fi][inside]
    CLIP = 170.0                                  # manifest clip ceiling
    frac_off = float((off > CLIP).mean())
    frac_on = float((on > CLIP).mean())
    assert frac_on < 0.5 * frac_off, \
        f"diffraction must materially reduce clipping: {frac_off:.3f} -> {frac_on:.3f}"
    assert frac_on < 0.06, f"{frac_on:.1%} of cells above the clip ceiling"
    assert float(np.median(on)) < 130.0, "median path loss implausible"
    assert np.all(on <= off + 1e-3)
    print(f"  [geom] no saturation constant: cells above the {CLIP:.0f} dB clip "
          f"ceiling {frac_off:.2%} -> {frac_on:.2%} with diffraction; "
          f"median {np.median(on):.1f} dB, p99.9 {np.percentile(on, 99.9):.1f} dB")


def _test_reciprocity():
    """Path loss must be (approximately) symmetric under swapping Tx and Rx.
    Exact reciprocity is broken only by which face the incidence angle is
    measured on, so a small residual is expected; a large one means the
    geometry is wrong."""
    g, inside, walk, cell = synth_floor_scene()
    sc = SceneV2(g, inside, cell, n_edges=0)
    fi = int(np.argmin(np.abs(sc.freqs - 3500.0)))
    a = (60.0, 60.0)
    b = (300.0, 190.0)
    ab = sc.pathloss_maps(a, with_diffraction=False)[fi, int(b[1]), int(b[0])]
    ba = sc.pathloss_maps(b, with_diffraction=False)[fi, int(a[1]), int(a[0])]
    assert abs(ab - ba) < 3.0, f"reciprocity violated: {ab:.2f} vs {ba:.2f} dB"
    print(f"  [geom] reciprocity: {ab:.2f} vs {ba:.2f} dB "
          f"(delta {abs(ab-ba):.2f})")


def run_tests():
    print("engine_v2 geometry self-tests")
    _test_crossing_extraction()
    _test_angle_of_incidence()
    _test_frequency_consistency()
    _test_edges_and_wedges()
    _test_diffraction_fills_shadow()
    _test_no_saturation_needed()
    _test_reciprocity()
    print("ALL GEOMETRY TESTS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        run_tests()
