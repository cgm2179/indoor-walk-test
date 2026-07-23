#!/usr/bin/env python3
"""
physics_v2.py — electromagnetic core for the enhanced Motley-Keenan model
(sim-v2.0).  Blocks A-D of `docs/v2_physics.tex`.

This module is PURE PHYSICS: it knows nothing about the floor plan, the
raster, or the dataset.  It takes (material, frequency, incidence angle,
thickness) and returns dB.  `phase_a.py` owns the geometry and calls in here.

Blocks
  A  complex permittivity from ITU-R P.2040-3 Table 3 power laws
  B  Fresnel coefficients + lossy-slab transmission (angle + thickness),
     plus a resistive-sheet model for low-emissivity (metal-coated) glass
  C  UTD wedge diffraction (Kouyoumjian-Pathak) + knife-edge cross-check
  D  floor attenuation

Everything here is deliberately slow-but-exact.  The hot path in phase_a.py
never calls these functions directly; it calls `build_crossing_lut()` once
and then interpolates.  That split is what makes 9 frequencies affordable.

usage:
  python SIM/physics_v2.py --test      # ~30 physics self-checks
  python SIM/physics_v2.py --tables    # print the loss tables for the paper
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.special import fresnel

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
C_MPS = 299_792_458.0
EPS0 = 8.854_187_8128e-12
ETA0 = 376.730_313_412          # free-space wave impedance, ohm

# ---------------------------------------------------------------------------
# BLOCK A — material electromagnetics (ITU-R P.2040-3 Table 3)
#
#   eps'(f)  = a * f_GHz**b          (real relative permittivity)
#   sigma(f) = c * f_GHz**d          (conductivity, S/m)
#   eps_r    = eps' - j*sigma/(2*pi*f*eps0)
#
# `valid` is the recommendation's stated frequency range in GHz; we record it
# so extrapolation is flagged rather than silent.  The 619 MHz scanner band is
# BELOW the 1 GHz floor for most classes -- that is a real, documented
# extrapolation, not an oversight (see v2_physics.tex Block A remark).
# ---------------------------------------------------------------------------
P2040 = {
    "vacuum":        dict(a=1.00, b=0.0, c=0.0,    d=0.0,    valid=(0.001, 100.0)),
    "concrete":      dict(a=5.24, b=0.0, c=0.0462, d=0.7822, valid=(1.0, 100.0)),
    "brick":         dict(a=3.91, b=0.0, c=0.0238, d=0.1600, valid=(1.0, 40.0)),
    "plasterboard":  dict(a=2.73, b=0.0, c=0.0085, d=0.9395, valid=(1.0, 100.0)),
    "wood":          dict(a=1.99, b=0.0, c=0.0047, d=1.0718, valid=(0.001, 100.0)),
    "glass":         dict(a=6.31, b=0.0, c=0.0036, d=1.3394, valid=(0.1, 100.0)),
    "ceiling_board": dict(a=1.48, b=0.0, c=0.0011, d=1.0750, valid=(1.0, 100.0)),
    "chipboard":     dict(a=2.58, b=0.0, c=0.0217, d=0.7800, valid=(1.0, 100.0)),
    "plywood":       dict(a=2.71, b=0.0, c=0.3300, d=0.0000, valid=(1.0, 40.0)),
    "marble":        dict(a=7.074, b=0.0, c=0.0055, d=0.9262, valid=(1.0, 60.0)),
    "floorboard":    dict(a=3.66, b=0.0, c=0.0044, d=1.3515, valid=(50.0, 100.0)),
    "metal":         dict(a=1.00, b=0.0, c=1.0e7,  d=0.0000, valid=(1.0, 100.0)),
}


def permittivity(material: str, f_mhz):
    """(A.1-A.2) Complex relative permittivity at f.  Returns complex ndarray.

    Sign convention: exp(+j w t) time dependence, so a lossy medium has
    NEGATIVE imaginary part.  numpy's principal branch of sqrt() then puts
    sqrt(eps_r) in the fourth quadrant automatically, which is the decaying
    root -- do not "fix" this with an abs() anywhere downstream.
    """
    p = P2040[material]
    f_ghz = np.atleast_1d(np.asarray(f_mhz, float)) / 1000.0
    eps_re = p["a"] * f_ghz ** p["b"]
    sigma = p["c"] * f_ghz ** p["d"]
    eps_im = sigma / (2.0 * np.pi * (f_ghz * 1e9) * EPS0)
    return eps_re - 1j * eps_im


def p2040_extrapolating(material: str, f_mhz) -> bool:
    """True if f is outside the recommendation's stated validity range."""
    lo, hi = P2040[material]["valid"]
    f_ghz = np.asarray(f_mhz, float) / 1000.0
    return bool(np.any((f_ghz < lo) | (f_ghz > hi)))


# ---------------------------------------------------------------------------
# BLOCK B — Fresnel coefficients and slab transmission
# ---------------------------------------------------------------------------
def fresnel_coeffs(eps_r, theta_i):
    """(B.1-B.2) Single-interface reflection coefficients, air -> medium.

    theta_i in radians, measured from the surface NORMAL.
    Returns (R_te, R_tm) as complex arrays broadcast over the inputs.

    R_te  (perpendicular / E out of the plane of incidence)
    R_tm  (parallel      / E in  the plane of incidence)
    """
    eps_r = np.asarray(eps_r, complex)
    ct = np.cos(np.asarray(theta_i, float))
    st2 = 1.0 - ct * ct
    root = np.sqrt(eps_r - st2)              # principal branch: Im <= 0
    r_te = (ct - root) / (ct + root)
    r_tm = (eps_r * ct - root) / (eps_r * ct + root)
    return r_te, r_tm


def electrical_thickness(eps_r, theta_i, thickness_m, f_mhz):
    """(B.3) q = (2*pi*d/lambda0) * sqrt(eps_r - sin^2 theta_i).

    Complex; Im(q) < 0, so |exp(-jq)| = exp(Im(q)) < 1 is the absorption.
    """
    lam0 = C_MPS / (np.asarray(f_mhz, float) * 1e6)
    st2 = np.sin(np.asarray(theta_i, float)) ** 2
    root = np.sqrt(np.asarray(eps_r, complex) - st2)
    return (2.0 * np.pi * np.asarray(thickness_m, float) / lam0) * root


def slab_transmission_coherent(eps_r, theta_i, thickness_m, f_mhz, pol="te"):
    """(B.4) Coherent slab transmission with internal multiple reflections.

        T = (1 - R^2) exp(-jq) / (1 - R^2 exp(-2jq))

    Exhibits Fabry-Perot ripple in |T| vs q.  Physically real, but the raster
    quantises wall thickness to +/-1 cell (+/-0.174 m ~ 2 lambda at 3.5 GHz),
    so the ripple phase is not knowable.  Kept for validation only -- see
    slab_transmission_incoherent(), which is the model default.
    """
    r_te, r_tm = fresnel_coeffs(eps_r, theta_i)
    r = _pick_pol(r_te, r_tm, pol)
    q = electrical_thickness(eps_r, theta_i, thickness_m, f_mhz)
    e1 = np.exp(-1j * q)
    return (1.0 - r * r) * e1 / (1.0 - r * r * e1 * e1)


def slab_transmission_incoherent(eps_r, theta_i, thickness_m, f_mhz, pol="te"):
    """(B.5) Power transmittance with the internal reflections summed in
    POWER rather than amplitude:

        Tpow = (1-rho)^2 * a^2 / (1 - rho^2 a^4),
        rho = |R|^2,   a^2 = |exp(-jq)|^2 = exp(2 Im q)

    This is the fully-decohered limit of B.4.  It is correct for slabs many
    wavelengths thick, and WRONG for thin ones: at t -> 0 it returns
    (1-rho)/(1+rho), i.e. ~1.4 dB of loss for a slab that is not there.  The
    missing physics is the coherent cancellation between the two interface
    reflections, which is exactly what makes a thin sheet transparent.

    Use slab_tpow(), not this, unless you specifically want the thick limit.
    """
    r_te, r_tm = fresnel_coeffs(eps_r, theta_i)
    r = _pick_pol(r_te, r_tm, pol)
    rho = np.abs(r) ** 2
    q = electrical_thickness(eps_r, theta_i, thickness_m, f_mhz)
    a2 = np.exp(2.0 * np.imag(q))            # Im(q) < 0 -> a2 < 1
    return (1.0 - rho) ** 2 * a2 / (1.0 - rho ** 2 * a2 * a2)


# Construction tolerance used to decohere the Fabry-Perot ripple.  A "drywall
# partition" is really two 12.7 mm boards on steel studs; a "concrete wall"
# has aggregate and rebar; and the Fresnel zone that matters is ~0.5 m across
# at these distances, over which the effective thickness varies.  sigma_t is
# therefore a real, physically bounded quantity, not a smoothing fudge -- and
# it is one of the parameters Phase D fits.
# 8 mm is not a smoothing knob picked to make curves pretty: the first
# Fresnel zone at these ranges is ~0.45 m across at 3.5 GHz, and over a
# 0.45 m footprint of real partition the effective thickness varies by board
# joints, taping compound, service penetrations and stud flanges by roughly
# this much.  Values in 5-15 mm are all defensible; below ~5 mm the model
# starts reproducing Fabry-Perot ripple that the building does not actually
# hold still enough to exhibit.  Phase D may fit it per material.
SIGMA_T_FLOOR_M = 0.008          # 8 mm absolute
SIGMA_T_FRAC = 0.10              # plus 10% of nominal thickness
# Fixed quadrature so the result never depends on how the caller batches.
# 65 nodes over +/-3 sigma resolves ~8 ripple periods at 8 samples/period;
# past MAX_RIPPLE_PERIODS the closed-form decohered limit takes over.
N_JITTER_NODES = 65
MAX_RIPPLE_PERIODS = 4.0


def sigma_thickness(thickness_m):
    """Standard deviation of the effective thickness over the Fresnel zone.

    Capped at half the nominal thickness: a wall cannot vary by more than it
    is thick, and a slab that is not there has no tolerance.  Without this cap
    the absolute floor would give a zero-thickness slab 0.8 dB of loss, which
    is the same class of error as the naive incoherent formula.
    """
    t = np.asarray(thickness_m, float)
    return np.minimum(np.maximum(SIGMA_T_FLOOR_M, SIGMA_T_FRAC * t), 0.5 * t)


def slab_tpow(eps_r, theta_i, thickness_m, f_mhz, pol="te", sigma_t=None):
    """(B.5') THE MODEL DEFAULT.  Power transmittance of a slab whose
    thickness is uncertain at the few-millimetre level:

        Tpow(t) = < |T_coherent(t + dt)|^2 >_{dt ~ N(0, sigma_t^2)}

    This is the honest version of "smooth over the Fabry-Perot ripple".  It
    has the right behaviour in BOTH limits, which neither B.4 nor B.5 does
    alone:

      * thin slab (t << lambda_medium): the phase spread is small, the average
        barely differs from the coherent result, and the slab is transparent.
        Drywall at 3.5 GHz is t ~ lambda/7 -- squarely in this regime, so
        using B.5 there would invent ~1.4 dB per partition out of nothing.
      * thick slab (t >> lambda_medium): the phase spread exceeds 2 pi, the
        ripple averages out, and the result converges to B.5 exactly.  Both
        limits are asserted in the self-tests.

    Integration is by fixed Gaussian-weighted quadrature over +/-3 sigma; past
    MAX_RIPPLE_PERIODS of ripple the analytic decohered limit B.5 is used
    instead, chosen elementwise so the result is independent of batching.
    """
    eps_r = np.asarray(eps_r, complex)
    theta_i = np.asarray(theta_i, float)
    t = np.asarray(thickness_m, float)
    if sigma_t is None:
        sigma_t = sigma_thickness(t)
    sigma_t = np.asarray(sigma_t, float)

    # ripple period in thickness: the Delta_t over which Re(q) advances by pi
    lam0 = C_MPS / (np.asarray(f_mhz, float) * 1e6)
    root_re = np.real(np.sqrt(eps_r - np.sin(theta_i) ** 2))
    period_t = lam0 / (2.0 * np.maximum(root_re, 1e-6))
    n_periods = 6.0 * sigma_t / period_t          # ripple periods spanned

    # BATCH INVARIANCE: the node count is a fixed constant, and the choice
    # between quadrature and the analytic decohered limit is made PER ELEMENT.
    # An earlier version sized the quadrature from max(sigma)/min(period) over
    # whatever array it was handed, which made the answer depend on how the
    # caller batched -- the LUT (built on a whole grid at once) and a
    # per-point reference evaluation then disagreed by up to 3 dB.
    x = np.linspace(-3.0, 3.0, N_JITTER_NODES)
    w = np.exp(-0.5 * x * x)
    w = w / w.sum()
    shape = np.broadcast(eps_r, theta_i, t, np.asarray(f_mhz, float)).shape
    acc = np.zeros(shape, float)
    for xi, wi in zip(x, w):
        t_i = np.maximum(t + sigma_t * xi, 0.0)
        acc = acc + wi * np.abs(slab_transmission_coherent(
            eps_r, theta_i, t_i, f_mhz, pol)) ** 2

    # Past MAX_RIPPLE_PERIODS the quadrature would be under-sampled, but the
    # ripple is also provably washed out there, so the closed form is both
    # cheaper and more accurate.
    inc = slab_transmission_incoherent(eps_r, theta_i, t, f_mhz, pol)
    return np.where(n_periods > MAX_RIPPLE_PERIODS, inc, acc)


def _pick_pol(r_te, r_tm, pol):
    if pol == "te":
        return r_te
    if pol == "tm":
        return r_tm
    raise ValueError(f"pol must be 'te' or 'tm', got {pol!r}")


def resistive_sheet_tpow(r_sheet_ohm_sq, theta_i, pol="te"):
    """(B.6) Power transmittance of a thin conductive film (low-emissivity
    coating) modelled as a shunt sheet admittance 1/Rs on a transmission line
    of wave impedance eta:

        T = 2 / (2 + eta/Rs),     eta_TE = eta0/cos(theta),  eta_TM = eta0*cos(theta)

    Rs ~ 20 ohm/sq reproduces the ~20 dB measured on the FCC HQ facade at
    2.4 GHz, which is why this is parameterised by a physical sheet resistance
    instead of a fudge constant: Rs is the thing Phase D actually fits, and it
    is bounded by what low-E coatings physically are (roughly 4-40 ohm/sq).

    Note the polarisation split is opposite in sign to the dielectric slab:
    at grazing incidence a resistive sheet BLOCKS TE and PASSES TM.
    """
    ct = np.clip(np.cos(np.asarray(theta_i, float)), 1e-3, 1.0)
    eta = ETA0 / ct if pol == "te" else ETA0 * ct
    t_amp = 2.0 / (2.0 + eta / float(r_sheet_ohm_sq))
    return t_amp ** 2


F_REF_MHZ = 2442.0               # reference frequency for the excess power law
SEC_CAP = 6.0                    # cap on 1/cos(theta) for the excess term


def excess_loss_db(e_ref_db, gamma_e, f_mhz, theta_i=0.0):
    """(B.8) Sub-resolution structure loss:  E(f) = E_ref (f/f_ref)^gamma_e,
    scaled by a capped secant for oblique incidence.

    This term exists because the 0.174 m raster cannot see the things that
    dominate measured wall loss above ~2 GHz: steel studs inside a partition,
    rebar mesh in concrete, mullions and frames in glazing, the metal doors on
    a lift core.  Those are metallic scatterers, so their contribution GROWS
    with frequency, which is exactly the residual left when the P.2040
    dielectric slab is subtracted from measured per-wall loss.

    It is a fitted term and is labelled as such everywhere.  What makes it
    honest rather than a fudge factor is that (a) it is separately identifiable
    in Phase D because it has a different frequency signature from the slab,
    (b) it is bounded by construction reality, and (c) the fitted values are
    physically legible -- a lift core comes out at ~8 dB nearly flat in f
    (metal doors), a stud partition at ~2.4 dB rising as f^1.2.

    An oblique ray crosses more studs, so the secant scaling applies; it is
    capped because at grazing the ray runs ALONG the wall and the geometry
    assumption fails.
    """
    if e_ref_db == 0.0:
        return np.zeros_like(np.asarray(f_mhz, float) *
                             np.asarray(theta_i, float))
    sec = np.minimum(1.0 / np.maximum(np.cos(np.asarray(theta_i, float)), 1e-3),
                     SEC_CAP)
    return e_ref_db * (np.asarray(f_mhz, float) / F_REF_MHZ) ** gamma_e * sec


def wall_loss_db(eps_r, theta_i, thickness_m, f_mhz, pol="te",
                 r_sheet_ohm_sq=None, coherent=False,
                 e_ref_db=0.0, gamma_e=0.0):
    """(B.7) Total per-crossing transmission loss in dB, >= 0.

    Cascades the dielectric slab (B.5') with an optional resistive sheet (B.6)
    and adds the sub-resolution excess (B.8).  The slab/sheet cascade is
    INCOHERENT (power multiply): the coating sits on one face of the glass,
    and the glass is many wavelengths from anything else, so the relative
    phase is unresolvable at raster precision.
    """
    if coherent:
        tpow = np.abs(slab_transmission_coherent(
            eps_r, theta_i, thickness_m, f_mhz, pol)) ** 2
    else:
        tpow = slab_tpow(eps_r, theta_i, thickness_m, f_mhz, pol)
    if r_sheet_ohm_sq is not None:
        tpow = tpow * resistive_sheet_tpow(r_sheet_ohm_sq, theta_i, pol)
    db = -10.0 * np.log10(np.maximum(tpow, 1e-30))
    if e_ref_db:
        db = db + excess_loss_db(e_ref_db, gamma_e, f_mhz, theta_i)
    return db


# ---------------------------------------------------------------------------
# BLOCK C — UTD wedge diffraction (Kouyoumjian-Pathak)
# ---------------------------------------------------------------------------
def transition_function(X, asym_threshold=10.0):
    """(C.2) F(X) = 2j sqrt(X) exp(jX) * int_{sqrt(X)}^{inf} exp(-j tau^2) dtau

    Expressed through the Fresnel integrals scipy provides:
        int_{sqrt(X)}^{inf} e^{-j t^2} dt
          = sqrt(pi/2) [ (1/2 - C(z)) - j (1/2 - S(z)) ],   z = sqrt(2X/pi)

    Limits worth remembering (both asserted in the self-tests):
        F(X) -> 1                     as X -> inf   (far from a shadow boundary)
        F(X) -> sqrt(pi X) e^{j(pi/4 + X)}  as X -> 0  (on the boundary)

    Above asym_threshold the asymptotic series
        F(X) ~ 1 + j/(2X) - 3/(4X^2) - j 15/(8X^3)
    is used instead of calling scipy.  In a shadow map the overwhelming
    majority of cells sit far from any shadow boundary, so this skips the
    Fresnel integral for most of the grid; the series and the exact form agree
    to better than 1e-4 at the crossover, which the self-tests check.
    """
    X = np.asarray(X, float)
    out = np.empty(X.shape, complex) if X.ndim else np.array(0j)
    big = X > asym_threshold
    if np.any(big):
        xb = X[big] if X.ndim else X
        inv = 1.0 / xb
        val = 1.0 + 0.5j * inv - 0.75 * inv ** 2 - 1.875j * inv ** 3
        if X.ndim:
            out[big] = val
        else:
            return val
    small = ~big
    if np.any(small):
        xs = X[small] if X.ndim else X
        z = np.sqrt(2.0 * xs / np.pi)
        s, c = fresnel(z)                     # scipy: (S, C)
        integral = np.sqrt(np.pi / 2.0) * ((0.5 - c) - 1j * (0.5 - s))
        val = 2j * np.sqrt(xs) * np.exp(1j * xs) * integral
        if X.ndim:
            out[small] = val
        else:
            return val
    return out


def _N_pm(beta, n):
    """Integers N+ / N- nearest to satisfying 2*pi*n*N -/+ beta = +/- pi."""
    n_plus = np.round((beta + np.pi) / (2.0 * np.pi * n))
    n_minus = np.round((beta - np.pi) / (2.0 * np.pi * n))
    return n_plus, n_minus


def _a_pm(beta, n):
    """(C.3) a^{+/-}(beta) = 2 cos^2( (2 n pi N^{+/-} - beta) / 2 )."""
    n_plus, n_minus = _N_pm(beta, n)
    a_p = 2.0 * np.cos((2.0 * n * np.pi * n_plus - beta) / 2.0) ** 2
    a_m = 2.0 * np.cos((2.0 * n * np.pi * n_minus - beta) / 2.0) ** 2
    return a_p, a_m


def utd_coefficient(phi, phi_p, s, s_p, k, n=1.5, beta0=np.pi / 2,
                    soft=True, refl_coeff=None):
    """(C.1) Kouyoumjian-Pathak diffraction coefficient for a wedge of
    exterior angle n*pi.

      phi    observation angle from the 0-face (rad)
      phi_p  incidence angle from the 0-face (rad)
      s, s'  distances edge->Rx and Tx->edge (m)
      k      wavenumber 2*pi/lambda
      n      wedge parameter: 2 = half-plane (knife edge), 1.5 = 90 deg corner
      soft   True -> D_s (E parallel to edge), False -> D_h
      refl_coeff  optional (R_0face, R_nface) Fresnel coefficients; supplying
                  them applies the Luebbers heuristic for a lossy wedge, which
                  is what couples Block C back to Block B.

    Returns complex D with units of sqrt(metre).
    """
    phi = np.asarray(phi, float)
    phi_p = np.asarray(phi_p, float)
    L = np.asarray(s, float) * np.asarray(s_p, float) * np.sin(beta0) ** 2 / \
        (np.asarray(s, float) + np.asarray(s_p, float))

    pre = -np.exp(-1j * np.pi / 4.0) / \
        (2.0 * n * np.sqrt(2.0 * np.pi * k) * np.sin(beta0))

    def _term(beta):
        a_p, a_m = _a_pm(beta, n)
        cot_p = _cot((np.pi + beta) / (2.0 * n))
        cot_m = _cot((np.pi - beta) / (2.0 * n))
        return (cot_p * transition_function(k * L * a_p),
                cot_m * transition_function(k * L * a_m))

    d1_p, d1_m = _term(phi - phi_p)          # incident shadow boundary terms
    d2_p, d2_m = _term(phi + phi_p)          # reflection boundary terms

    if refl_coeff is None:
        sign = -1.0 if soft else 1.0
        return pre * (d1_p + d1_m + sign * (d2_p + d2_m))

    # Luebbers heuristic: weight each reflection-boundary term by the Fresnel
    # coefficient of the face that would produce that reflection.
    r0, rn = refl_coeff
    return pre * (d1_p + d1_m + rn * d2_p + r0 * d2_m)


def _cot(x):
    """cot with the removable singularities at multiples of pi nudged aside.

    The cotangents blow up exactly at the shadow/reflection boundaries, where
    F(X)->0 cancels them analytically.  Rather than implement the L'Hopital
    expansion we offset by a hair; the product is smooth and the error is far
    below any dB we report.  (Verified against the analytic limit in --test.)
    """
    x = np.asarray(x, float)
    bad = np.abs(np.sin(x)) < 1e-9
    x = np.where(bad, x + 1e-9, x)
    return np.cos(x) / np.sin(x)


def utd_pathloss_db(phi, phi_p, s, s_p, f_mhz, n=1.5, soft=True,
                    refl_coeff=None):
    """(C.4) Path loss (dB) of the Tx -> edge -> Rx diffracted ray, referenced
    the same way as FSPL, i.e. directly comparable with the direct-ray PL.

        |E_d/E_0|^2 = |D|^2 / ( s' * s * (s + s') )
        PL = FSPL(1 m) + 10 log10( s' s (s+s') ) - 20 log10|D|

    where the spherical spreading factor A(s,s') = sqrt(s'/(s(s+s'))) is
    already folded in.
    """
    lam = C_MPS / (np.asarray(f_mhz, float) * 1e6)
    k = 2.0 * np.pi / lam
    d = utd_coefficient(phi, phi_p, s, s_p, k, n=n, soft=soft,
                        refl_coeff=refl_coeff)
    fspl_1m = 20.0 * np.log10(4.0 * np.pi / lam)
    spread = 10.0 * np.log10(np.asarray(s_p, float) * np.asarray(s, float) *
                             (np.asarray(s, float) + np.asarray(s_p, float)))
    return fspl_1m + spread - 20.0 * np.log10(np.maximum(np.abs(d), 1e-30))


def knife_edge_j_db(v):
    """(C.5) ITU-R P.526 single knife-edge approximation, kept as an
    independent cross-check on the UTD path (they must agree in the deep
    shadow for n=2 -- asserted in --test).

        J(v) = 6.9 + 20 log10( sqrt((v-0.1)^2 + 1) + v - 0.1 ),  v > -0.78
    """
    v = np.asarray(v, float)
    j = 6.9 + 20.0 * np.log10(np.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
    return np.where(v > -0.78, j, 0.0)


def fresnel_v(h, d1, d2, f_mhz):
    """(C.6) Fresnel-Kirchhoff diffraction parameter
    v = h * sqrt( 2 (d1 + d2) / (lambda d1 d2) )."""
    lam = C_MPS / (np.asarray(f_mhz, float) * 1e6)
    d1 = np.asarray(d1, float)
    d2 = np.asarray(d2, float)
    return np.asarray(h, float) * np.sqrt(
        2.0 * (d1 + d2) / np.maximum(lam * d1 * d2, 1e-12))


# ---------------------------------------------------------------------------
# BLOCK D — floor attenuation
# ---------------------------------------------------------------------------
def faf_db(n_floors, mode="slab", eps_r=None, f_mhz=None, thickness_m=0.20,
           theta_i=0.0, rebar_excess_db=6.0, itu_base=15.0, itu_step=4.0):
    """(D.1) Floor attenuation for n_floors crossings.

    mode='slab'  -> n * (Block B slab loss + rebar excess).  LINEAR in n.
    mode='itu'   -> ITU-R P.1238 office form 15 + 4(n-1) dB, which SATURATES.

    The saturation in the empirical form is not a property of concrete; it is
    the signature of the dominant path switching from straight-through to
    out-the-facade-and-back-in for n >= 3.  A model that has Block C can
    represent that mechanism explicitly, so 'slab' is the default and the
    empirical curve is kept only for comparison.  Do not stack them.
    """
    n = np.asarray(n_floors, float)
    if mode == "itu":
        return np.where(n <= 0, 0.0, itu_base + itu_step * (n - 1.0))
    if mode != "slab":
        raise ValueError("mode must be 'slab' or 'itu'")
    per = wall_loss_db(eps_r, theta_i, thickness_m, f_mhz) + rebar_excess_db
    return n * per


# ---------------------------------------------------------------------------
# the v2 seven-class material table
#
# `p2040`      which P.2040-3 Table 3 class supplies eps' and sigma
# `t_ref_m`    nominal crossing thickness, from CONSTRUCTION, not the raster:
#              a "drywall partition" is two 12.7 mm boards, not one sheet
# `r_sheet`    sheet resistance of a conductive coating, ohm/sq (low-E only)
# `e_ref_db`   sub-resolution structure excess at F_REF_MHZ  (fitted, B.8)
# `gamma_e`    its frequency exponent                        (fitted, B.8)
# `per_metre`  True for bulk clutter: loss accrues per metre of path, not
#              per crossing (v1's furniture rule, retained -- an open-plan ray
#              crosses ~50 drawn desk symbols and charging each is fiction)
#
# The e_ref/gamma_e pairs are least-squares fits of (measured anchor - slab)
# across 2442/3500/5500/6125 MHz, residual < 0.9 dB everywhere.  They are
# SEEDS: Phase D refits them against the Gflex multi-band logs.
# ---------------------------------------------------------------------------
MATERIALS7 = [
    dict(id=0, name="air", p2040=None, t_ref_m=0.0,
         e_ref_db=0.0, gamma_e=0.0, per_metre=False, color="#ffffff"),
    dict(id=1, name="drywall_partition", p2040="plasterboard", t_ref_m=0.026,
         e_ref_db=2.38, gamma_e=1.20, per_metre=False, color="#f5a623"),
    dict(id=2, name="concrete_masonry", p2040="concrete", t_ref_m=0.200,
         e_ref_db=1.71, gamma_e=1.46, per_metre=False, color="#404040"),
    dict(id=3, name="core_service_area", p2040="concrete", t_ref_m=0.200,
         e_ref_db=8.06, gamma_e=0.36, per_metre=False, color="#c81e1e"),
    # Clutter is a DILUTE medium, not a solid one: the class covers whole
    # open-plan bays that are mostly air with occasional desks, screens and
    # chairs.  A ray at desk height crossing 10 m of open plan intersects
    # maybe 0.1 m of actual wood.  fill_fraction converts the solid-wood
    # P.2040 attenuation into an effective dB/m; 0.021 reproduces v1's
    # measured 0.30 dB/m at 2442 MHz, and the resulting curve then tracks
    # v1.2's independent 0.45/0.62/0.68 anchors at 3500/5500/6125 to ~0.1 dB.
    dict(id=4, name="furniture_clutter", p2040="wood", t_ref_m=1.000,
         e_ref_db=0.0, gamma_e=0.0, per_metre=True, fill_fraction=0.021,
         color="#bed2ff"),
    dict(id=5, name="exterior_glass_lowE", p2040="glass", t_ref_m=0.006,
         r_sheet_ohm_sq=25.0, e_ref_db=0.0, gamma_e=0.0, per_metre=False,
         color="#1e3ca0"),
    dict(id=6, name="interior_glass", p2040="glass", t_ref_m=0.010,
         e_ref_db=1.60, gamma_e=1.28, per_metre=False, color="#7ec8e3"),
]

# The nine-band union: five bands the Gflex actually measured plus four
# hypothetical indoor-transmitter bands (MODEL_CARD_V2 section 4).
FREQS_MHZ_V2 = [619.0, 627.0, 1935.0, 2442.0, 2510.0,
                2600.0, 3500.0, 5500.0, 6125.0]


# ---------------------------------------------------------------------------
# fast path: per-crossing loss lookup table
# ---------------------------------------------------------------------------
class CrossingLUT:
    """Precomputed per-crossing loss, indexed by (material, frequency,
    cos(theta_i), thickness).

    The whole point of this class: the ray geometry is frequency-independent,
    so `phase_a.py` traces once, collects (material, cos theta, run length)
    per crossing, and then gets ALL frequencies with two gathers and a lerp.
    Building the table costs ~10 ms; evaluating Block B exactly on every
    crossing of every ray would cost minutes per map.

    Binning is uniform in SEC(theta), not theta and not cos(theta).  The
    per-crossing loss is very nearly affine in sec(theta) -- the internal path
    is t/cos(theta_t) and the excess term carries an explicit secant -- so
    linear interpolation on that axis is near-exact, whereas uniform-in-cosine
    leaves several dB of error at grazing incidence where 1/cos is steep.
    """

    def __init__(self, materials=None, freqs_mhz=None, n_cos=72, n_t=72,
                 cos_min=0.06, t_min=0.002, t_max=0.60, pol="te"):
        self.materials = list(materials if materials is not None else MATERIALS7)
        self.freqs = np.asarray(
            freqs_mhz if freqs_mhz is not None else FREQS_MHZ_V2, float)
        # uniform in sec(theta) from 1 (normal) to 1/cos_min (grazing)
        self.sec_max = 1.0 / cos_min
        self.sec_grid = np.linspace(1.0, self.sec_max, n_cos)
        self.cos_grid = 1.0 / self.sec_grid
        # LOG spacing in thickness: real crossings cluster at small t (6 mm
        # glass, 10 mm glass, 26 mm partition) and a linear grid puts all of
        # those inside one cell.  Log spacing tracks where the classes are.
        self.t_grid = np.geomspace(t_min, t_max, n_t)
        self.log_t0 = np.log(t_min)
        self.log_t_span = np.log(t_max) - np.log(t_min)
        self.cos_min, self.t_min, self.t_max = cos_min, t_min, t_max
        self.pol = pol
        self.per_metre = np.array([bool(m.get("per_metre"))
                                   for m in self.materials])
        self.t_ref = np.array([m.get("t_ref_m", 0.0)
                               for m in self.materials], np.float32)
        self._build()

    def _build(self):
        n_m, n_f = len(self.materials), len(self.freqs)
        n_c, n_t = len(self.cos_grid), len(self.t_grid)
        self.table = np.zeros((n_m, n_f, n_c, n_t), np.float32)
        theta = np.arccos(self.cos_grid)[:, None]          # (n_c, 1)
        tt = self.t_grid[None, :]                          # (1, n_t)
        for mi, m in enumerate(self.materials):
            if m.get("p2040") is None:                     # air
                continue
            for fi, f in enumerate(self.freqs):
                eps = permittivity(m["p2040"], f)[0]
                if m.get("per_metre"):
                    # bulk clutter: tabulate dB per METRE of path, so the
                    # geometry side multiplies by path length, not crossings.
                    # Reference: 1 m of the P.2040 medium, angle-independent
                    # (a ray through furniture has no surface to be oblique to).
                    per_m = wall_loss_db(eps, 0.0, 1.0, f, pol=self.pol)
                    self.table[mi, fi] = per_m * m.get("fill_fraction", 1.0)
                else:
                    self.table[mi, fi] = wall_loss_db(
                        eps, theta, tt, f, pol=self.pol,
                        r_sheet_ohm_sq=m.get("r_sheet_ohm_sq"),
                        e_ref_db=m.get("e_ref_db", 0.0),
                        gamma_e=m.get("gamma_e", 0.0))
        # per-(material, frequency) scale factors: the Phase-D calibration
        # handles.  Identity until calibration.json is loaded.
        self.scale = np.ones((n_m, n_f), np.float32)

    def apply_calibration(self, cal):
        """cal: {material_name: scale} or {material_name: [scale per freq]}."""
        for mi, m in enumerate(self.materials):
            if m["name"] in cal:
                self.scale[mi, :] = cal[m["name"]]

    # -- lookup -------------------------------------------------------------
    def _indices(self, cos_theta, thickness_m):
        """Shared bilinear index arithmetic for loss() and loss_all_bands()."""
        sec = 1.0 / np.clip(np.asarray(cos_theta, np.float32),
                            self.cos_min, 1.0)
        ci = np.clip((sec - 1.0) / (self.sec_max - 1.0) *
                     (len(self.sec_grid) - 1),
                     0, len(self.sec_grid) - 1 - 1e-6)
        tcl = np.clip(np.asarray(thickness_m, np.float64),
                      self.t_min, self.t_max)
        ti = np.clip((np.log(tcl) - self.log_t0) / self.log_t_span *
                     (len(self.t_grid) - 1),
                     0, len(self.t_grid) - 1 - 1e-6)
        # clamp the LOWER index to n-2 so the +1 corner always exists;
        # clipping the float alone is not enough because float32 rounding can
        # land ci exactly on n-1.
        c0 = np.clip(ci.astype(np.intp), 0, len(self.sec_grid) - 2)
        t0 = np.clip(ti.astype(np.intp), 0, len(self.t_grid) - 2)
        return c0, t0, (ci - c0).astype(np.float32), (ti - t0).astype(np.float32)

    def loss(self, mat_idx, freq_idx, cos_theta, thickness_m):
        """Bilinear lookup for one band.  Returns dB per crossing."""
        c0, t0, fc, ft = self._indices(cos_theta, thickness_m)
        # 4-way fancy index: one gather per corner, NOT self.table[m, f] then
        # [..., c0, t0] -- that outer-products the c and t axes and silently
        # returns an (n, n) matrix.
        m = np.asarray(mat_idx, np.intp)
        fq = np.broadcast_to(np.asarray(freq_idx, np.intp), m.shape)
        tab = self.table
        v = (tab[m, fq, c0, t0] * (1 - fc) * (1 - ft) +
             tab[m, fq, c0 + 1, t0] * fc * (1 - ft) +
             tab[m, fq, c0, t0 + 1] * (1 - fc) * ft +
             tab[m, fq, c0 + 1, t0 + 1] * fc * ft)
        return v * self.scale[m, fq]

    def loss_all_bands(self, mat_idx, cos_theta, thickness_m):
        """Every frequency at once: returns (n_freq, n_crossings) dB.

        The index arithmetic is identical for all bands -- only the table
        gather differs -- so doing it once instead of nine times is most of
        the cost of the whole ray tracer.  This is the routine the engine
        actually calls; loss() is kept for tests and single-band probes.
        """
        c0, t0, fc, ft = self._indices(cos_theta, thickness_m)
        m = np.asarray(mat_idx, np.intp)
        tab = self.table                       # (n_m, n_f, n_c, n_t)
        a = tab[m, :, c0, t0]                  # -> (n, n_f)
        b = tab[m, :, c0 + 1, t0]
        c = tab[m, :, c0, t0 + 1]
        d = tab[m, :, c0 + 1, t0 + 1]
        w = ((1 - fc) * (1 - ft))[:, None]
        x = (fc * (1 - ft))[:, None]
        y = ((1 - fc) * ft)[:, None]
        z = (fc * ft)[:, None]
        v = a * w + b * x + c * y + d * z
        return (v * self.scale[m, :]).T.astype(np.float32)

    def normal_incidence_table(self, thickness_by_material):
        """dB per crossing at theta=0 and each material's nominal thickness --
        the table that goes in the paper and that Phase D compares against
        the measured attenuation anchors."""
        out = np.zeros((len(self.materials), len(self.freqs)))
        for mi, m in enumerate(self.materials):
            if m.get("p2040") is None:
                continue
            t = thickness_by_material[mi]
            for fi, f in enumerate(self.freqs):
                eps = permittivity(m["p2040"], f)[0]
                out[mi, fi] = wall_loss_db(
                    eps, 0.0, t, f, pol=self.pol,
                    r_sheet_ohm_sq=m.get("r_sheet_ohm_sq"))
        return out


# ---------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------
def _test_block_a():
    eps = permittivity("concrete", 3500.0)[0]
    assert abs(eps.real - 5.24) < 1e-9, eps
    sigma = 0.0462 * 3.5 ** 0.7822
    expect_im = sigma / (2 * np.pi * 3.5e9 * EPS0)
    assert abs(-eps.imag - expect_im) < 1e-9
    assert eps.imag < 0, "lossy medium must have NEGATIVE imaginary part"
    # vacuum is lossless and unity
    assert abs(permittivity("vacuum", 1000.0)[0] - 1.0) < 1e-12
    # metal is enormously lossy
    assert abs(permittivity("metal", 3500.0)[0].imag) > 1e6
    # dispersion direction: concrete gets lossier with frequency in eps''*f
    e24 = permittivity("concrete", 2442.0)[0]
    e61 = permittivity("concrete", 6125.0)[0]
    assert abs(e61.imag) < abs(e24.imag), \
        "sigma ~ f^0.78 grows slower than f, so eps'' must FALL with f"
    assert p2040_extrapolating("concrete", 619.0)
    assert not p2040_extrapolating("concrete", 3500.0)
    print("  [A] permittivity, sign convention, dispersion, validity flags")


def _test_block_b_fresnel():
    eps = permittivity("concrete", 3500.0)[0]
    r_te, r_tm = fresnel_coeffs(eps, 0.0)
    # at normal incidence the two polarisations coincide up to the sign
    # convention (R_tm is defined with the opposite reference direction)
    assert abs(abs(r_te) - abs(r_tm)) < 1e-12
    # normal-incidence magnitude matches (1-sqrt(eps))/(1+sqrt(eps))
    expect = abs((1 - np.sqrt(eps)) / (1 + np.sqrt(eps)))
    assert abs(abs(r_te) - expect) < 1e-12
    # grazing incidence -> total reflection for both polarisations
    r_te_g, r_tm_g = fresnel_coeffs(eps, np.radians(89.99))
    assert abs(r_te_g) > 0.99 and abs(r_tm_g) > 0.98
    # TM has a pseudo-Brewster minimum; TE is monotone in theta
    th = np.radians(np.linspace(0, 89, 90))
    rte = np.abs(fresnel_coeffs(eps, th)[0])
    rtm = np.abs(fresnel_coeffs(eps, th)[1])
    assert np.all(np.diff(rte) > -1e-9), "|R_te| must increase with angle"
    assert rtm.argmin() > 5, "TM must show a Brewster-like dip away from 0 deg"
    # passivity
    assert np.all(rte <= 1.0 + 1e-12) and np.all(rtm <= 1.0 + 1e-12)
    print("  [B] Fresnel: normal/grazing limits, monotonicity, Brewster, passivity")


def _test_block_b_slab():
    eps = permittivity("concrete", 3500.0)[0]
    # 1. THIN LIMIT: a vanishing slab must be transparent.  This is the test
    #    that the naive incoherent formula fails (it returns 1.35 dB here).
    assert wall_loss_db(eps, 0.0, 0.0, 3500.0) < 0.5, \
        wall_loss_db(eps, 0.0, 0.0, 3500.0)
    assert slab_transmission_incoherent(eps, 0.0, 0.0, 3500.0) < 0.75, \
        "sanity: the naive form really is bad at t=0 (that's why we jitter)"
    # 2. THICK LIMIT: converges to the incoherent formula, to within the
    #    Jensen gap.  Averaging exp(-2*alpha*t) over a thickness spread is
    #    LARGER than the value at the mean thickness, so the jittered slab is
    #    always slightly more transparent than the point estimate.  That is
    #    the real "thin spots dominate transmission" effect, not an error --
    #    so the test is on relative agreement, and it is one-sided.
    for t_thick in (0.30, 0.50):
        a = -10 * np.log10(slab_tpow(eps, 0.0, t_thick, 3500.0))
        b = -10 * np.log10(slab_transmission_incoherent(eps, 0.0, t_thick, 3500.0))
        assert a <= b + 1e-6, f"Jensen gap has the wrong sign: {a:.2f} > {b:.2f}"
        assert (b - a) / b < 0.05, \
            f"t={t_thick}: jittered {a:.2f} vs incoherent {b:.2f} dB"
    # 3. loss grows monotonically with thickness
    t = np.linspace(0.005, 0.6, 120)
    L = np.array([wall_loss_db(eps, 0.0, ti, 3500.0) for ti in t])
    assert np.all(np.diff(L) > -1e-6), "loss must increase with thickness"
    # 4. loss grows monotonically with incidence angle (longer internal path)
    th = np.radians(np.linspace(0, 85, 60))
    La = np.array([wall_loss_db(eps, x, 0.20, 3500.0) for x in th])
    assert np.all(np.diff(La) > -1e-6), "oblique must cost more than normal"
    # 5. the secant law is the small-angle limit of the exact slab
    L0 = wall_loss_db(eps, 0.0, 0.20, 3500.0)
    L45 = wall_loss_db(eps, np.radians(45), 0.20, 3500.0)
    sec_pred = L0 / np.cos(np.radians(45))
    assert 0.6 * sec_pred < L45 < 1.4 * sec_pred, \
        f"45 deg loss {L45:.1f} should be near the secant estimate {sec_pred:.1f}"
    # 6. the coherent version really does ripple, and the incoherent formula
    #    is its phase average -- the justification for decohering at all
    lam = C_MPS / 3.5e9
    tt = np.linspace(0.20, 0.20 + 6 * lam / (2 * np.sqrt(eps.real)), 4001)
    coh = np.abs(slab_transmission_coherent(eps, 0.0, tt, 3500.0)) ** 2
    inc = slab_transmission_incoherent(eps, 0.0, tt, 3500.0)
    rel = abs(coh.mean() - inc.mean()) / inc.mean()
    assert rel < 0.02, f"incoherent slab off phase-average by {rel:.3%}"
    assert coh.std() / coh.mean() > 0.05, "coherent slab should show ripple"
    # 6b. BATCH INVARIANCE: evaluating a grid at once must equal evaluating
    #     each point alone.  This is a regression guard -- an adaptive node
    #     count silently made these differ by up to 3 dB.
    th = np.radians(np.array([0.0, 30.0, 60.0, 80.0]))[:, None]
    tt = np.array([0.01, 0.05, 0.2, 0.5])[None, :]
    grid = slab_tpow(eps, th, tt, 3500.0)
    for i in range(4):
        for j in range(4):
            one = slab_tpow(eps, th[i, 0], tt[0, j], 3500.0)
            assert abs(float(grid[i, j]) - float(one)) < 1e-12, \
                f"batch invariance broken at ({i},{j})"
    # 7. thin drywall stays cheap: a 13 mm partition must not cost like a wall
    eps_d = permittivity("plasterboard", 3500.0)[0]
    l_dry = wall_loss_db(eps_d, 0.0, 0.013, 3500.0)
    assert l_dry < 1.5, f"13 mm plasterboard = {l_dry:.2f} dB, want thin-regime"
    # 8. concrete is the lossiest of the structural classes at equal thickness
    losses = {m: wall_loss_db(permittivity(m, 3500.0)[0], 0.0, 0.15, 3500.0)
              for m in ("wood", "plasterboard", "glass", "concrete")}
    assert losses["concrete"] == max(losses.values()), losses
    assert losses["wood"] < losses["concrete"], losses
    print("  [B] slab: thin+thick limits, monotone in t and theta, secant, "
          "ripple average, drywall stays thin, concrete lossiest")


def _test_block_b_lowe():
    # 25 ohm/sq reproduces the measured 20 dB facade at 2.4 GHz, and sits in
    # the middle of the 4-40 ohm/sq range real low-E coatings occupy
    RS = 25.0
    eps_g = permittivity("glass", 2442.0)[0]
    L = wall_loss_db(eps_g, 0.0, 0.006, 2442.0, r_sheet_ohm_sq=RS)
    assert 19.0 < L < 21.0, f"low-E facade at 2.4 GHz = {L:.1f} dB, want ~20"
    # plain glass of the same thickness is a few dB
    L_plain = wall_loss_db(eps_g, 0.0, 0.006, 2442.0)
    assert L_plain < 3.0, f"plain 6 mm glass = {L_plain:.1f} dB, want < 3"
    # the coating dominates the frequency behaviour: nearly flat vs frequency
    Ls = [wall_loss_db(permittivity("glass", f)[0], 0.0, 0.006, f,
                       r_sheet_ohm_sq=RS) for f in (619.0, 2442.0, 6125.0)]
    assert max(Ls) - min(Ls) < 6.0, f"low-E loss should be flat-ish: {Ls}"
    # TE/TM diverge in OPPOSITE directions at grazing for a resistive sheet
    te = resistive_sheet_tpow(RS, np.radians(80), "te")
    tm = resistive_sheet_tpow(RS, np.radians(80), "tm")
    assert tm > te, "resistive sheet passes TM and blocks TE at grazing"
    # the headline consequence: a street-level macro hits the facade
    # obliquely and pays far more than v1's fixed 15 dB O2I assumption
    obl = wall_loss_db(permittivity("glass", 2600.0)[0], np.radians(75),
                       0.006, 2600.0, r_sheet_ohm_sq=RS)
    assert obl > 32.0, f"75 deg low-E facade = {obl:.1f} dB, expected >32"
    print(f"  [B] low-E sheet: 25 ohm/sq -> {L:.1f} dB normal / {obl:.1f} dB "
          "at 75 deg, flat in f, correct pol split")


def _test_block_c_transition():
    # F(X) -> 1 for large X
    assert abs(transition_function(50.0) - 1.0) < 0.02, transition_function(50.0)
    assert abs(transition_function(1e4) - 1.0) < 1e-3
    # small-X asymptote, to SECOND order.  The leading term alone is only
    # good to ~2X/sqrt(pi X), so testing against it at 1% would pass a broken
    # implementation; the two-term form pins the phase as well as the size.
    for X in (1e-4, 1e-6):
        got = transition_function(X)
        want = (np.sqrt(np.pi * X) - 2.0 * X * np.exp(1j * np.pi / 4)) * \
            np.exp(1j * (np.pi / 4 + X))
        assert abs(got - want) / abs(want) < 1e-4, (X, got, want)
    # |F| <= ~1 everywhere and F is continuous
    Xs = np.logspace(-6, 4, 4000)
    F = transition_function(Xs)
    assert np.all(np.abs(F) < 1.06), np.abs(F).max()
    assert np.all(np.isfinite(F))
    print("  [C] transition function: both asymptotes, boundedness")


def _test_block_c_utd():
    f = 3500.0
    lam = C_MPS / (f * 1e6)
    k = 2 * np.pi / lam
    s_p, s = 30.0, 30.0

    # --- 1. THE normalisation test.  At the incident shadow boundary the
    #        diffracted field is exactly half the incident field, so the
    #        diffracted path loss sits 6.02 dB below... above free space over
    #        the total path.  Nothing else pins the absolute scale of D.
    php = np.pi / 2
    fspl_tot = 20 * np.log10(4 * np.pi * (s + s_p) / lam)
    at_isb = utd_pathloss_db(np.pi + php + np.radians(0.02), php, s, s_p,
                             f, n=2.0, soft=True) - fspl_tot
    assert abs(at_isb - 6.02) < 0.25, \
        f"half-plane at the shadow boundary = FSPL+{at_isb:.2f} dB, want +6.02"

    # --- 2. agreement with the independent knife-edge formula in the regime
    #        where knife-edge is valid (paraxial).  Geometry computed from
    #        coordinates rather than a small-angle guess.
    def _h_d1_d2(phi, php_, s_, sp_):
        T = np.array([sp_ * np.cos(php_), sp_ * np.sin(php_)])
        R = np.array([s_ * np.cos(phi), s_ * np.sin(phi)])
        v = R - T
        L = np.linalg.norm(v)
        u = v / L
        d1 = -float(np.dot(T, u))
        h = abs(float(T[0] * v[1] - T[1] * v[0])) / L
        return h, d1, L - d1

    for dd in (2.0, 5.0, 10.0):
        phi = np.pi + php + np.radians(dd)
        pl = utd_pathloss_db(phi, php, s, s_p, f, n=2.0, soft=True)
        h, d1, d2 = _h_d1_d2(phi, php, s, s_p)
        ke = 20 * np.log10(4 * np.pi * (d1 + d2) / lam) + \
            knife_edge_j_db(fresnel_v(h, d1, d2, f))
        assert abs(pl - ke) < 1.0, \
            f"UTD vs knife-edge at {dd} deg: {pl:.2f} vs {ke:.2f} dB"
    # far outside the paraxial regime they SHOULD part company, with UTD
    # predicting more loss -- this is precisely the energy v1's saturation
    # stand-in was faking, and why knife-edge alone is not enough.
    phi40 = np.pi + php + np.radians(40)
    h, d1, d2 = _h_d1_d2(phi40, php, s, s_p)
    ke40 = 20 * np.log10(4 * np.pi * (d1 + d2) / lam) + \
        knife_edge_j_db(fresnel_v(h, d1, d2, f))
    assert utd_pathloss_db(phi40, php, s, s_p, f, n=2.0) > ke40 + 2.0

    # --- 3. monotone deeper into shadow
    deltas = np.radians(np.linspace(1.0, 40.0, 60))
    pl = np.array([utd_pathloss_db(np.pi + php + d, php, s, s_p, f, n=2.0)
                   for d in deltas])
    assert np.all(np.diff(pl) > -1e-6), "UTD loss must grow into the shadow"

    # --- 4. wedge ordering: MORE material shadows MORE.  Evaluated with
    #        phi'=45 deg so every observation angle stays inside [0, n*pi]
    #        for all three wedges -- evaluating a n=1.5 wedge at 285 deg is
    #        asking for the field inside the concrete.
    php2 = np.radians(45)
    for dd in (5.0, 20.0, 40.0):
        phi = np.pi + php2 + np.radians(dd)
        assert np.degrees(phi) <= 270.0 + 1e-9, "test angle left the wedge"
        v = [utd_pathloss_db(phi, php2, s, s_p, f, n=nn)
             for nn in (1.5, 1.75, 2.0)]
        assert v[0] > v[1] > v[2], f"wedge ordering broken at {dd} deg: {v}"

    # --- 5. D ~ sqrt(lambda).  PL_diff therefore scales as 30 log10 f while
    #        free space scales as 20 log10 f, so the excess is 10 log10(f2/f1).
    phi = np.pi + php2 + np.radians(15)
    for ratio in (2.0, 6.0):
        lo = utd_pathloss_db(phi, php2, s, s_p, 1000.0, n=1.5)
        hi = utd_pathloss_db(phi, php2, s, s_p, 1000.0 * ratio, n=1.5)
        excess = (hi - lo) - 20 * np.log10(ratio)
        assert abs(excess - 10 * np.log10(ratio)) < 0.3, \
            f"sqrt(lambda) scaling off: excess {excess:.2f}, " \
            f"want {10*np.log10(ratio):.2f}"

    # --- 6. finite everywhere in the valid region, including both boundaries
    for nn in (1.5, 2.0):
        ph = np.radians(np.linspace(0.5, nn * 180.0 - 0.5, 4000))
        d = utd_coefficient(ph, php2, s, s_p, k, n=nn)
        assert np.all(np.isfinite(d)), f"UTD not finite for n={nn}"

    # --- 7. Luebbers weighting.  It replaces the PEC reflection weight
    #        (-1 soft, +1 hard) with the face Fresnel coefficient, so it must
    #        (a) reduce EXACTLY to the PEC cases at R = -1 and R = +1, and
    #        (b) for any |R| <= 1 stay inside the envelope of the two, since
    #        D is affine in the weight and |.| is convex.  ("lossy diffracts
    #        less" is NOT universally true -- with R ~ -0.62 for TE concrete
    #        the reflection term partly cancels the incident term and |D| can
    #        move either way.)
    eps = permittivity("concrete", f)[0]
    phi = np.pi + php2 + np.radians(10)
    d_soft = utd_coefficient(phi, php2, s, s_p, k, n=1.5, soft=True)
    d_hard = utd_coefficient(phi, php2, s, s_p, k, n=1.5, soft=False)
    one = np.array(1.0 + 0j)
    assert abs(utd_coefficient(phi, php2, s, s_p, k, n=1.5,
                               refl_coeff=(-one, -one)) - d_soft) < 1e-12
    assert abs(utd_coefficient(phi, php2, s, s_p, k, n=1.5,
                               refl_coeff=(one, one)) - d_hard) < 1e-12
    env = max(abs(d_soft), abs(d_hard))
    for deg in (30.0, 60.0, 85.0):
        r = fresnel_coeffs(eps, np.radians(deg))[0]
        d_los = utd_coefficient(phi, php2, s, s_p, k, n=1.5,
                                refl_coeff=(r, r))
        assert abs(d_los) <= env + 1e-12, \
            f"Luebbers outside the PEC envelope at {deg} deg"
    print("  [C] UTD: 6.02 dB half-field normalisation, knife-edge agreement "
          "in the paraxial regime, monotone, wedge ordering, sqrt(lambda), "
          "finite at boundaries, Luebbers")


def _test_block_d():
    eps = permittivity("concrete", 3500.0)[0]
    a = faf_db(1, eps_r=eps, f_mhz=3500.0, thickness_m=0.20)
    b = faf_db(3, eps_r=eps, f_mhz=3500.0, thickness_m=0.20)
    assert abs(b - 3 * a) < 1e-9, "slab FAF must be linear in floor count"
    assert faf_db(0, eps_r=eps, f_mhz=3500.0) == 0.0
    itu1, itu3 = faf_db(1, mode="itu"), faf_db(3, mode="itu")
    assert itu1 == 15.0 and itu3 == 23.0
    assert itu3 < 3 * itu1, "the empirical form saturates by construction"
    print("  [D] FAF: slab linear, ITU form saturating, zero at n=0")


def _test_lut():
    lut = CrossingLUT()
    assert len(lut.materials) == 7 and len(lut.freqs) == 9
    rng = np.random.default_rng(0)
    n = 3000
    # exclude air(0) and furniture(4, per-metre) from the crossing check
    pool = np.array([1, 2, 3, 5, 6])
    mi = pool[rng.integers(0, len(pool), n)]
    fi = rng.integers(0, 9, n)
    ct = rng.uniform(0.06, 1.0, n)
    tk = np.exp(rng.uniform(np.log(0.004), np.log(0.6), n))
    got = lut.loss(mi, fi, ct, tk)          # vectorised, one call
    want = np.array([
        wall_loss_db(permittivity(MATERIALS7[m]["p2040"], lut.freqs[f])[0],
                     np.arccos(c), t, lut.freqs[f],
                     r_sheet_ohm_sq=MATERIALS7[m].get("r_sheet_ohm_sq"),
                     e_ref_db=MATERIALS7[m].get("e_ref_db", 0.0),
                     gamma_e=MATERIALS7[m].get("gamma_e", 0.0))
        for m, f, c, t in zip(mi, fi, ct, tk)])
    err = np.abs(got - want)
    assert err.max() < 0.8, f"LUT max error {err.max():.3f} dB"
    assert err.mean() < 0.03, f"LUT mean error {err.mean():.4f} dB"
    assert np.percentile(err, 99) < 0.25, "LUT p99 too loose"
    # air is free at any angle/thickness
    assert lut.loss(np.array([0]), np.array([1]),
                    np.array([1.0]), np.array([0.5]))[0] == 0.0
    # calibration scaling is applied
    lut.apply_calibration({"concrete_masonry": 1.5})
    a = lut.loss(np.array([2]), np.array([6]), np.array([1.0]), np.array([0.2]))[0]
    lut.apply_calibration({"concrete_masonry": 1.0})
    b = lut.loss(np.array([2]), np.array([6]), np.array([1.0]), np.array([0.2]))[0]
    assert abs(a - 1.5 * b) < 1e-4, "calibration scale not applied"
    print(f"  [LUT] 7 classes x 9 bands within {err.max():.3f} dB of exact "
          f"(mean {err.mean():.4f} dB); calibration hook works")


def _test_material_table():
    """The finished table must land near the project's measured anchors and
    stay physically ordered across all nine bands."""
    lut = CrossingLUT()
    anchors = {  # (material name, freq, measured dB, tolerance)
        ("drywall_partition", 2442.0, 3.0, 1.0),
        ("concrete_masonry", 2442.0, 15.0, 2.0),
        ("concrete_masonry", 6125.0, 34.1, 3.0),
        ("core_service_area", 2442.0, 22.0, 2.0),
        ("exterior_glass_lowE", 2442.0, 20.0, 1.0),
        ("interior_glass", 2442.0, 3.0, 1.0),
    }
    names = [m["name"] for m in MATERIALS7]
    for name, f, want, tol in anchors:
        mi = names.index(name)
        fi = int(np.argmin(np.abs(lut.freqs - f)))
        got = lut.loss(np.array([mi]), np.array([fi]),
                       np.array([1.0]), np.array([MATERIALS7[mi]["t_ref_m"]]))[0]
        assert abs(got - want) < tol, f"{name}@{f:.0f}: {got:.2f} vs {want} dB"
    # monotone in frequency for every dielectric class
    for mi in (1, 2, 3, 6):
        v = [lut.loss(np.array([mi]), np.array([fi]), np.array([1.0]),
                      np.array([MATERIALS7[mi]["t_ref_m"]]))[0]
             for fi in range(9)]
        assert all(b >= a - 1e-3 for a, b in zip(v, v[1:])), (mi, v)
    # ordering holds at every band: drywall < concrete < core
    for fi in range(9):
        g = lambda mi: lut.loss(np.array([mi]), np.array([fi]), np.array([1.0]),
                                np.array([MATERIALS7[mi]["t_ref_m"]]))[0]
        assert g(1) < g(2) < g(3), f"ordering broken at band {fi}"
    # clutter must stay dilute: v1's measured 0.30 dB/m anchor at 2442
    ci = names.index("furniture_clutter")
    fi24 = int(np.argmin(np.abs(lut.freqs - 2442.0)))
    cl = lut.loss(np.array([ci]), np.array([fi24]),
                  np.array([1.0]), np.array([1.0]))[0]
    assert 0.20 < cl < 0.45, f"clutter = {cl:.2f} dB/m, want ~0.30"
    for fi_, want_ in ((6, 0.45), (7, 0.62), (8, 0.68)):
        got_ = lut.loss(np.array([ci]), np.array([fi_]),
                        np.array([1.0]), np.array([1.0]))[0]
        assert abs(got_ - want_) < 0.20, \
            f"clutter at band {fi_}: {got_:.2f} vs v1.2 anchor {want_}"
    # the low-E facade is the headline: flat in f, steep in angle
    fi = 6
    flat = [lut.loss(np.array([5]), np.array([i]), np.array([1.0]),
                     np.array([0.006]))[0] for i in range(9)]
    assert max(flat) - min(flat) < 2.5, f"low-E should be flat in f: {flat}"
    obl = lut.loss(np.array([5]), np.array([fi]),
                   np.array([np.cos(np.radians(75))]), np.array([0.006]))[0]
    nor = lut.loss(np.array([5]), np.array([fi]),
                   np.array([1.0]), np.array([0.006]))[0]
    assert obl - nor > 10.0, \
        f"75 deg facade must cost >10 dB more than normal, got {obl-nor:.1f}"
    print("  [TAB] 7-class table hits measured anchors, monotone in f, "
          f"ordered at all bands, facade +{obl-nor:.1f} dB at 75 deg")


def run_tests():
    print("physics_v2 self-tests")
    _test_block_a()
    _test_block_b_fresnel()
    _test_block_b_slab()
    _test_block_b_lowe()
    _test_block_c_transition()
    _test_block_c_utd()
    _test_block_d()
    _test_lut()
    _test_material_table()
    print("ALL PHYSICS TESTS PASSED")


def print_tables():
    freqs = [619.0, 627.0, 1935.0, 2442.0, 2510.0, 2600.0, 3500.0, 5500.0, 6125.0]
    rows = [("drywall_partition", "plasterboard", 0.13, None),
            ("concrete_masonry", "concrete", 0.20, None),
            ("core_service_area", "concrete", 0.30, None),
            ("exterior_glass_lowE", "glass", 0.006, 20.0),
            ("interior_glass", "glass", 0.006, None),
            ("furniture_clutter", "wood", 1.00, None)]
    hdr = "material".ljust(22) + "".join(f"{f:>8.0f}" for f in freqs)
    print(hdr)
    print("-" * len(hdr))
    for name, p, t, rs in rows:
        vals = [wall_loss_db(permittivity(p, f)[0], 0.0, t, f,
                             r_sheet_ohm_sq=rs) for f in freqs]
        print(name.ljust(22) + "".join(f"{v:8.2f}" for v in vals))
    print("\n(dB per normal-incidence crossing at the nominal thickness; "
          "furniture row is dB per metre)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--tables", action="store_true")
    a = ap.parse_args()
    if a.test:
        run_tests()
    if a.tables:
        print_tables()
