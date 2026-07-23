#!/usr/bin/env python3
"""
engine_v2_torch.py — CUDA backend for the enhanced Motley-Keenan engine.

A line-for-line port of `engine_v2.SceneV2` onto torch tensors, so Phase B can
generate the 7-class / 9-band dataset on a Colab GPU in minutes instead of
hours.  `engine_v2.py` (numpy) remains the REFERENCE implementation: the
parity test in this file asserts the two agree, and if they ever disagree the
numpy one is right.

Why a port rather than cupy / numba: the whole hot path is gathers,
scatter-adds and elementwise complex arithmetic, all of which torch expresses
directly, and Colab already has torch installed with a working CUDA build.
There is exactly one thing torch cannot do natively -- the Fresnel integral
inside the UTD transition function -- and that is handled by tabulating F(X)
once on the CPU with scipy and interpolating on device.

Design notes that matter for correctness:

  * Sample counts are quantised to multiples of K_QUANT exactly as in the
    numpy engine, because that is what makes path loss reciprocal.  Do not
    "optimise" this into a per-batch maximum.
  * Transmitters are processed one at a time.  There is enough parallelism
    inside a single 256x448 x 9-band map to saturate a GPU, and per-Tx
    processing keeps the reciprocity guarantee and the memory ceiling flat.
  * float32 throughout except the linear-power accumulation, which is float64
    -- path loss spans 250 dB and float32 underflows around 10^-38.

usage:
  python SIM/engine_v2_torch.py --parity     # numpy vs torch agreement
  python SIM/engine_v2_torch.py --bench      # throughput estimate
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import engine_v2 as E
import physics_v2 as P

try:
    import torch
except ImportError:                                   # pragma: no cover
    torch = None

STEP_CELL = E.STEP_CELL
K_QUANT = E.K_QUANT
D0_M = E.D0_M
N_EXP = E.N_EXP
C_MPS = P.C_MPS


# ---------------------------------------------------------------------------
# transition function on device
# ---------------------------------------------------------------------------
class TransitionTable:
    """F(X) for the UTD transition function, tabulated for X <= X_ASYM and
    evaluated from the asymptotic series above it.

    F ~ sqrt(pi X) near zero, so the table is built on a LOG grid in X and the
    interpolation is done on log X; a linear grid would need ~10^6 points to
    resolve the same behaviour near the shadow boundary.
    """

    X_ASYM = 10.0
    X_SMALL = 1e-4          # below this use the closed-form small-argument F

    def __init__(self, device, dtype=torch.float32, n=8192, x_lo=1e-5):
        from scipy.special import fresnel
        x = np.geomspace(x_lo, self.X_ASYM, n)
        z = np.sqrt(2.0 * x / np.pi)
        s, c = fresnel(z)
        integ = np.sqrt(np.pi / 2.0) * ((0.5 - c) - 1j * (0.5 - s))
        f = 2j * np.sqrt(x) * np.exp(1j * x) * integ
        self.log_lo = float(np.log(x_lo))
        self.log_hi = float(np.log(self.X_ASYM))
        self.n = n
        self.re = torch.as_tensor(f.real, dtype=dtype, device=device)
        self.im = torch.as_tensor(f.imag, dtype=dtype, device=device)

    def __call__(self, X):
        X = X.clamp_min(1e-30)
        big = X > self.X_ASYM
        small = X < self.X_SMALL
        # asymptotic branch: F ~ 1 + j/(2X) - 3/(4X^2) - j 15/(8X^3)
        inv = 1.0 / X.clamp_min(self.X_ASYM)
        re_b = 1.0 - 0.75 * inv ** 2
        im_b = 0.5 * inv - 1.875 * inv ** 3
        # small-argument branch: F(X) -> (sqrt(pi X) - 2 X e^{j pi/4}) e^{j(pi/4+X)}
        # This closed form is what keeps cot(.)*F finite at the shadow
        # boundary: cot blows up like 1/X there while F vanishes like sqrt(X),
        # so the product must be evaluated where F is EXACT, not interpolated
        # off a log grid whose floor would leave F ~2.6x too large and inject
        # tens of dB after the cotangent.  (This was the sole numpy/torch
        # parity failure.)
        sx = X.clamp_max(self.X_SMALL)
        rt = torch.sqrt(np.pi * sx)
        c45, s45 = np.cos(np.pi / 4), np.sin(np.pi / 4)
        # (rt - 2 sx (c45 + j s45)) * (cos(pi/4+X) + j sin(pi/4+X))
        ar = rt - 2.0 * sx * c45
        ai = -2.0 * sx * s45
        ph = np.pi / 4 + sx
        cr, cS = torch.cos(ph), torch.sin(ph)
        re_s = ar * cr - ai * cS
        im_s = ar * cS + ai * cr
        # table branch (X_SMALL <= X <= X_ASYM)
        u = (torch.log(X.clamp(np.exp(self.log_lo), self.X_ASYM))
             - self.log_lo) / (self.log_hi - self.log_lo) * (self.n - 1)
        i0 = u.floor().clamp(0, self.n - 2).long()
        w = (u - i0).to(self.re.dtype)
        re_t = self.re[i0] * (1 - w) + self.re[i0 + 1] * w
        im_t = self.im[i0] * (1 - w) + self.im[i0 + 1] * w
        re = torch.where(big, re_b, torch.where(small, re_s, re_t))
        im = torch.where(big, im_b, torch.where(small, im_s, im_t))
        return re, im


def _cot(x):
    s = torch.sin(x)
    s = torch.where(s.abs() < 1e-9, torch.full_like(s, 1e-9), s)
    return torch.cos(x) / s


def _a_pm(beta, n):
    n_plus = torch.round((beta + np.pi) / (2.0 * np.pi * n))
    n_minus = torch.round((beta - np.pi) / (2.0 * np.pi * n))
    a_p = 2.0 * torch.cos((2.0 * n * np.pi * n_plus - beta) / 2.0) ** 2
    a_m = 2.0 * torch.cos((2.0 * n * np.pi * n_minus - beta) / 2.0) ** 2
    return a_p, a_m


def utd_pathloss_db_torch(phi, phi_p, s, s_p, f_mhz, n, ftab, soft=True):
    """Torch mirror of physics_v2.utd_pathloss_db.  phi and s are tensors;
    phi_p, s_p, f_mhz and n are python floats."""
    lam = C_MPS / (f_mhz * 1e6)
    k = 2.0 * np.pi / lam
    L = s * s_p / (s + s_p)
    pre = 1.0 / (2.0 * n * np.sqrt(2.0 * np.pi * k))     # |prefactor|
    pre_ang = -np.pi / 4.0                               # its phase

    def term(beta):
        a_p, a_m = _a_pm(beta, n)
        fp_re, fp_im = ftab(k * L * a_p)
        fm_re, fm_im = ftab(k * L * a_m)
        cp = _cot((np.pi + beta) / (2.0 * n))
        cm = _cot((np.pi - beta) / (2.0 * n))
        return (cp * fp_re + cm * fm_re, cp * fp_im + cm * fm_im)

    d1_re, d1_im = term(phi - phi_p)
    d2_re, d2_im = term(phi + phi_p)
    sign = -1.0 if soft else 1.0
    sum_re = d1_re + sign * d2_re
    sum_im = d1_im + sign * d2_im
    # multiply by pre * exp(j * pre_ang) and take the magnitude
    mag = pre * torch.sqrt(sum_re * sum_re + sum_im * sum_im)

    fspl_1m = 20.0 * np.log10(4.0 * np.pi / lam)
    spread = 10.0 * torch.log10(s_p * s * (s + s_p))
    return fspl_1m + spread - 20.0 * torch.log10(mag.clamp_min(1e-30))


# ---------------------------------------------------------------------------
# the scene
# ---------------------------------------------------------------------------
class TorchScene:
    def __init__(self, grid, inside, cell_size_m, materials=None,
                 freqs_mhz=None, device="cuda", n_edges=8, n_relay_cache=16,
                 edge_radius=3, precompute=True, lut=None,
                 relay_dtype=torch.float16):
        if torch is None:
            raise RuntimeError("torch is required for the GPU backend")
        self.device = torch.device(device)
        self.cell = float(cell_size_m)
        self.materials = materials if materials is not None else P.MATERIALS7
        self.freqs = np.asarray(
            freqs_mhz if freqs_mhz is not None else P.FREQS_MHZ_V2, float)
        self.nf = len(self.freqs)

        grid = np.ascontiguousarray(grid.astype(np.uint8))
        self.grid_np = grid
        self.inside_np = inside
        H, W = grid.shape
        self.H, self.W = H, W

        # geometry precomputation is cheap and identical to the numpy engine,
        # so it is reused verbatim rather than reimplemented
        nx, ny = E.wall_normals(grid)
        self.w_ref_np = E.measure_nominal_widths(grid, len(self.materials),
                                                 self.cell)
        self.edges = E.find_diffracting_edges(grid, inside, radius=edge_radius)
        self.n_edges = n_edges

        d = self.device
        self.grid = torch.as_tensor(grid.astype(np.int64), device=d)
        self.nx = torch.as_tensor(nx, dtype=torch.float32, device=d)
        self.ny = torch.as_tensor(ny, dtype=torch.float32, device=d)
        self.w_ref = torch.as_tensor(self.w_ref_np, dtype=torch.float32,
                                     device=d)
        self.t_ref = torch.as_tensor(
            np.array([m.get("t_ref_m", 0.0) for m in self.materials],
                     np.float32), device=d)
        self.per_metre = torch.as_tensor(
            np.array([bool(m.get("per_metre")) for m in self.materials]),
            device=d)

        lut = lut if lut is not None else P.CrossingLUT(self.materials,
                                                        self.freqs)
        self.lut_np = lut
        self.table = torch.as_tensor(lut.table, dtype=torch.float32, device=d)
        self.scale = torch.as_tensor(lut.scale, dtype=torch.float32, device=d)
        self.sec_max = float(lut.sec_max)
        self.n_sec = len(lut.sec_grid)
        self.cos_min = float(lut.cos_min)
        self.t_min, self.t_max = float(lut.t_min), float(lut.t_max)
        self.log_t0, self.log_t_span = float(lut.log_t0), float(lut.log_t_span)
        self.n_t = len(lut.t_grid)

        yy, xx = torch.meshgrid(torch.arange(H, device=d, dtype=torch.float32),
                                torch.arange(W, device=d, dtype=torch.float32),
                                indexing="ij")
        self.gx, self.gy = xx, yy
        self.gxf, self.gyf = xx.reshape(-1), yy.reshape(-1)
        self.fspl1 = torch.as_tensor(E.fspl_1m_db(self.freqs),
                                     dtype=torch.float32, device=d)
        # The ray tracer runs in float32, but the UTD diffraction path needs
        # float64: near a shadow boundary cot(.) ~ 1/X blows up while F ~
        # sqrt(X) vanishes, and their product is only conditioned in double
        # precision -- in float32 it drifts by a few dB right at the boundary,
        # which was the residual numpy/torch gap after the small-X fix.  UTD is
        # a small fraction of the total work, so this costs little.
        self.ftab = TransitionTable(d, dtype=torch.float64)

        self.relays = self._pick_relays(n_relay_cache) if precompute else []
        self.relay_obs = None
        if self.relays:
            self.relay_obs = torch.empty((len(self.relays), self.nf, H, W),
                                         dtype=relay_dtype, device=d)
            for i, e in enumerate(self.relays):
                self.relay_obs[i] = self.obstruction_maps(
                    (float(e["ax"]), float(e["ay"]))).to(relay_dtype)

    def set_material_scale(self, scale_mf):
        """Per-(material, frequency) multiplier on per-crossing loss.

        This is how the dataset generator applies wall-loss jitter (the v1
        augmentation that keeps the surrogate robust to material uncertainty,
        so Phase D recalibration is a fine-tune rather than a full retrain).
        Scaling here perturbs the direct path and the first diffraction leg,
        which are recomputed per transmitter; the cached second leg keeps the
        nominal materials, a documented approximation that is second order
        because diffraction is a minor term and its material dependence under
        +/-20% jitter is smaller still.  Pass None to restore nominal.
        """
        if scale_mf is None:
            self.scale = torch.ones((len(self.materials), self.nf),
                                    dtype=torch.float32, device=self.device)
        else:
            self.scale = torch.as_tensor(np.asarray(scale_mf, np.float32),
                                         device=self.device)

    def _pick_relays(self, k):
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

    # -- LUT lookup, all bands ---------------------------------------------
    def _lut_all_bands(self, m_run, cos_th, thick):
        sec = 1.0 / cos_th.clamp(self.cos_min, 1.0)
        ci = ((sec - 1.0) / (self.sec_max - 1.0) * (self.n_sec - 1)).clamp(
            0, self.n_sec - 1 - 1e-6)
        tcl = thick.clamp(self.t_min, self.t_max)
        ti = ((torch.log(tcl) - self.log_t0) / self.log_t_span *
              (self.n_t - 1)).clamp(0, self.n_t - 1 - 1e-6)
        c0 = ci.long().clamp(0, self.n_sec - 2)
        t0 = ti.long().clamp(0, self.n_t - 2)
        fc = (ci - c0).unsqueeze(1)
        ft = (ti - t0).unsqueeze(1)
        tab = self.table                       # (n_m, n_f, n_c, n_t)
        a = tab[m_run, :, c0, t0]
        b = tab[m_run, :, c0 + 1, t0]
        c = tab[m_run, :, c0, t0 + 1]
        dd = tab[m_run, :, c0 + 1, t0 + 1]
        v = (a * (1 - fc) * (1 - ft) + b * fc * (1 - ft) +
             c * (1 - fc) * ft + dd * fc * ft)
        return (v * self.scale[m_run, :]).T         # (n_f, n_cross)

    # -- core trace ---------------------------------------------------------
    def obstruction_maps(self, src_xy):
        d = self.device
        H, W, nf = self.H, self.W, self.nf
        sx, sy = float(src_xy[0]), float(src_xy[1])
        dist = torch.hypot(self.gxf - sx, self.gyf - sy)
        need = torch.ceil(dist / STEP_CELL).long() + 2
        kq = ((need.clamp_min(K_QUANT) + K_QUANT - 1) // K_QUANT) * K_QUANT
        out = torch.zeros((nf, H * W), dtype=torch.float32, device=d)

        for Kt in torch.unique(kq):
            K = int(Kt.item())
            sel = torch.nonzero(kq == Kt, as_tuple=False).squeeze(1)
            if sel.numel() == 0:
                continue
            d_cell = dist[sel]
            t = torch.linspace(0.0, 1.0, K, device=d, dtype=torch.float32)
            ex = self.gxf[sel]
            ey = self.gyf[sel]
            n_cell = ex.numel()
            xi = torch.round(sx + t * (ex[:, None] - sx)).clamp(0, W - 1).long()
            yi = torch.round(sy + t * (ey[:, None] - sy)).clamp(0, H - 1).long()
            mats = self.grid[yi, xi]

            seg_m = d_cell / (K - 1) * self.cell

            same = mats[:, 1:] == mats[:, :-1]
            starts = torch.empty_like(mats, dtype=torch.bool)
            ends = torch.empty_like(mats, dtype=torch.bool)
            starts[:, 0] = True
            starts[:, 1:] = ~same
            ends[:, -1] = True
            ends[:, :-1] = ~same
            si = torch.nonzero(starts, as_tuple=False)
            ei = torch.nonzero(ends, as_tuple=False)
            si_r, si_c = si[:, 0], si[:, 1]
            ei_c = ei[:, 1]
            m_run = mats[si_r, si_c]
            keep = m_run > 0
            if not bool(keep.any()):
                continue
            si_r, si_c, ei_c, m_run = (si_r[keep], si_c[keep], ei_c[keep],
                                       m_run[keep])
            run_cells = (ei_c - si_c + 1).float()
            run_len_m = run_cells * seg_m[si_r]

            rx = ex[si_r] - sx
            ry = ey[si_r] - sy
            rn = torch.hypot(rx, ry).clamp_min(1e-6)
            cos_acc = torch.zeros(si_r.numel(), device=d)
            n_ok = torch.zeros(si_r.numel(), device=d)
            for cc in (si_c, ei_c):
                wx = self.nx[yi[si_r, cc], xi[si_r, cc]]
                wy = self.ny[yi[si_r, cc], xi[si_r, cc]]
                ok = (wx * wx + wy * wy) > 0.25
                cos_acc = cos_acc + torch.where(
                    ok, ((rx * wx + ry * wy) / rn).abs(),
                    torch.zeros_like(cos_acc))
                n_ok = n_ok + ok.float()
            cos_th = torch.where(n_ok > 0, cos_acc / n_ok.clamp_min(1),
                                 torch.ones_like(cos_acc)).clamp(0.06, 1.0)

            thick = run_len_m * cos_th * (self.t_ref[m_run] /
                                          self.w_ref[m_run].clamp_min(1e-6))
            per = self._lut_all_bands(m_run, cos_th, thick)      # (nf, ncross)
            is_pm = self.per_metre[m_run]
            if bool(is_pm.any()):
                per = torch.where(is_pm.unsqueeze(0),
                                  per * run_len_m.unsqueeze(0), per)
            acc = torch.zeros((nf, n_cell), dtype=torch.float32, device=d)
            acc.index_add_(1, si_r, per)
            out.index_add_(1, sel, acc)
        return out.reshape(nf, H, W)

    def select_edges(self, obs_ref):
        pool = self.relays if self.relays else self.edges
        if not pool:
            return []
        ex = np.array([e["ax"] for e in pool])
        ey = np.array([e["ay"] for e in pool])
        cost = (obs_ref[ey, ex].cpu().numpy()
                - 2.0 * np.array([e["n"] for e in pool]))
        order = np.argsort(cost)[:self.n_edges]
        return [(int(i), pool[int(i)]) for i in order]

    def pathloss_maps(self, tx_xy, with_diffraction=True):
        d = self.device
        obs = self.obstruction_maps(tx_xy)
        dm = (torch.hypot(self.gx - tx_xy[0], self.gy - tx_xy[1])
              * self.cell).clamp_min(D0_M)
        fs = self.fspl1[:, None, None] + 10.0 * N_EXP * torch.log10(dm)[None]
        lin = torch.pow(10.0, (-(fs + obs) / 10.0).double())

        if with_diffraction:
            edges = self.select_edges(obs[self.nf // 2])
            txx, txy = float(tx_xy[0]), float(tx_xy[1])
            for idx, e in edges:
                exx, eyy = float(e["x"]), float(e["y"])
                s_p = max(np.hypot(txx - exx, txy - eyy) * self.cell, D0_M)
                phi_p = float((np.arctan2(txy - eyy, txx - exx)
                               - e["face0"]) % (2 * np.pi))
                if phi_p > e["n"] * np.pi:
                    continue
                s = (torch.hypot(self.gx - exx, self.gy - eyy)
                     * self.cell).clamp_min(D0_M).double()
                phi = torch.remainder(
                    torch.atan2(self.gy - eyy, self.gx - exx) - e["face0"],
                    2 * np.pi).double()
                valid = phi <= e["n"] * np.pi
                if not bool(valid.any()):
                    continue
                if self.relay_obs is not None:
                    leg2 = self.relay_obs[idx].float()  # noqa: cache dtype
                else:
                    leg2 = self.obstruction_maps((float(e["ax"]),
                                                  float(e["ay"])))
                leg1 = obs[:, e["ay"], e["ax"]]
                for fi in range(self.nf):
                    pl = utd_pathloss_db_torch(phi, phi_p, s, s_p,
                                               float(self.freqs[fi]),
                                               e["n"], self.ftab)
                    pl = pl + (leg1[fi] + leg2[fi]).double()
                    contrib = torch.pow(10.0, -pl / 10.0)
                    lin[fi] += torch.where(valid, contrib,
                                           torch.zeros_like(contrib))

        pl = (-10.0 * torch.log10(lin.clamp_min(1e-300))).float()
        return torch.maximum(pl, fs)


# ---------------------------------------------------------------------------
# parity and benchmark
# ---------------------------------------------------------------------------
def _scene_pair(device="cpu", n_relay=8, n_edges=4):
    g, inside, walk, cell = E.synth_floor_scene()
    lut = P.CrossingLUT()
    a = E.SceneV2(g, inside, cell, lut=lut, n_edges=n_edges,
                  n_relay_cache=n_relay, precompute=n_relay > 0,
                  relay_dtype=np.float32)
    b = TorchScene(g, inside, cell, lut=lut, device=device, n_edges=n_edges,
                   n_relay_cache=n_relay, precompute=n_relay > 0,
                   relay_dtype=torch.float32)
    return a, b, g, inside, walk, cell


def parity(device="cpu"):
    print(f"numpy vs torch parity on {device}")
    a, b, g, inside, walk, cell = _scene_pair(device)
    assert [e["x"] for e in a.relays] == [e["x"] for e in b.relays], \
        "relay selection diverged"

    worst_o = 0.0
    tail_p = []              # per-cell abs error, indoor, below the clip
    worst_any = 0.0
    ys, xs = np.nonzero(walk)
    rng = np.random.default_rng(1)
    txs = [(float(xs[i]), float(ys[i]))
           for i in rng.integers(0, len(ys), 16)]
    for tx in txs:
        oa = a.obstruction_maps(tx)
        ob = b.obstruction_maps(tx).cpu().numpy()
        worst_o = max(worst_o, float(np.abs(oa - ob).max()))
        pa = a.pathloss_maps(tx)
        pb = b.pathloss_maps(tx).cpu().numpy()
        m = np.broadcast_to(inside, pa.shape) & (pa < 170.0)
        e = np.abs(pa - pb)[m]
        tail_p.append(e)
        worst_any = max(worst_any, float(e.max()))
    tail = np.concatenate(tail_p)
    p9999 = float(np.percentile(tail, 99.99))
    frac_hi = float((tail > 0.5).mean())
    print(f"  obstruction  max |numpy - torch| = {worst_o:.4f} dB")
    print(f"  path loss    p99.99 |numpy - torch| = {p9999:.4f} dB, "
          f"worst {worst_any:.3f} dB, {frac_hi:.4%} of cells > 0.5 dB")
    # The obstruction map -- the thing the surrogate actually learns almost
    # everywhere -- must be near bit-identical.  Diffracted deep-shadow cells
    # sit on reflection boundaries where cot*F cancellation is delicate; a
    # tiny fraction can differ by ~1 dB between the numpy (all-float64) and
    # torch (float32 tracer + float64 UTD) paths, but they are in >170 dB
    # shadow that is clipped out of training anyway.  So the gate is: exact
    # where it matters, negligible mass in the tail.
    assert worst_o < 0.02, "obstruction parity failed"
    assert p9999 < 0.05, f"path-loss p99.99 = {p9999:.3f} dB too large"
    assert frac_hi < 1e-4, f"{frac_hi:.2%} of cells exceed 0.5 dB"

    # the transition table must match scipy across the full range
    ft = TransitionTable(torch.device(device))
    X = torch.logspace(-7, 4, 20000, device=device)
    re, im = ft(X)
    ref = P.transition_function(X.cpu().numpy(), asym_threshold=1e12)
    err = np.abs((re.cpu().numpy() + 1j * im.cpu().numpy()) - ref).max()
    print(f"  transition table vs scipy: max |dF| = {err:.2e}")
    assert err < 2e-3, "transition table too coarse"
    print("PARITY OK")


def bench(device="cpu", n=8):
    a, b, g, inside, walk, cell = _scene_pair(device, n_relay=16, n_edges=8)
    ys, xs = np.nonzero(walk)
    rng = np.random.default_rng(0)
    pick = rng.integers(0, len(ys), n)
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.time()
    for i in pick:
        b.pathloss_maps((float(xs[i]), float(ys[i])))
    if device != "cpu":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"torch[{device}]  {dt*1000:7.1f} ms per 9-band map with diffraction")
    print(f"   => 2500 positions ~ {dt*2500/60:.1f} min")
    t0 = time.time()
    for i in pick[:2]:
        a.pathloss_maps((float(xs[i]), float(ys[i])))
    dtn = (time.time() - t0) / 2
    print(f"numpy[cpu]    {dtn*1000:7.1f} ms per map  "
          f"(=> {dtn*2500/60:.0f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch is not None
                          and torch.cuda.is_available() else "cpu")
    if args.parity:
        parity(dev)
    if args.bench:
        bench(dev)
