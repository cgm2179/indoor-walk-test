# Model Card — 7th-Floor Path-Loss Surrogate (sim-v2.0) · PLANNED

> **STATUS: PLANNED / not yet built.** The deployed, validated model is v1
> (`sim-v1.1`, test RMSE 4.68 dB — see `MODEL_CARD.md`). This card is the
> design + workflow for the next generation, so it can be reviewed and built
> incrementally. Nothing here is trained yet. Mechanical retrain steps live
> in `retrain_for_physics_training_map_v2.md`; this card is the *what and why*.

Project name for this generation: **Indoor Walk Test v3** (the physical walk
was v1, the surrogate-in-browser was v2/sim-v1.x, this enhanced-physics +
outdoor-foundation generation is v3 of the overall effort, carrying the
model version `sim-v2.0`).

---

## 1. Headline: what changes v1 → v2

v1 is a *basic* Motley-Keenan surrogate (free-space + fixed per-wall dB +
clutter, with an empirical saturation standing in for diffraction). v2 is an
**enhanced / modified Motley-Keenan** model — the mathematical framework best
suited to process, calibrate, and predict the multi-band indoor+outdoor data
the PCTEL Gflex captures.

| Aspect | v1 (sim-v1.1, deployed) | v2 (sim-v2.0, planned) |
|---|---|---|
| Wall loss vs frequency | global multiplier ×1.0/1.15/1.30/1.40 | **per-material frequency exponent** `L_w(f)=L_ref·(f/f_ref)^γ` |
| Angle of incidence | ignored (assumes 90° crossing) | **oblique crossings cost more** (see §2.2) |
| Wall thickness | binary thick/thin → concrete/drywall | **loss scales with rasterized run length** |
| Diffraction | saturation stand-in (40→90 dB cap) | **explicit knife-edge / UTD fill** (spec 12.3-2) |
| Floors | single floor | **FAF term** scaffolded for multi-floor / vertical |
| Exterior glass | 3 dB (assumed clear) | **20 dB (measured low-E/coated FCC facade)** |
| Interior/lunch glass | folded into exterior | **separate class, ~3 dB normal glass** |
| Frequencies | 4 indoor-Tx anchors | **9-anchor union** incl. the 5 scanner bands |
| Calibration | none (literature values) | **Gflex least-squares fit** (Phase D formalized) |
| Scope | indoor only | indoor + **outdoor sandbox foundation** (§7) |
| Reference | Motley-Keenan (1988) | enhanced MK — **IEEE Xplore doc 8016211** + §2 |

Reference: the enhanced multi-wall formulation is based on IEEE Xplore
document **8016211** (paywalled — pull the exact equations from the paper
before implementing §2.1–2.5). The specific enhancements below (angle of
incidence, thickness scaling, frequency exponent, FAF, diffraction merge) are
the standard levers that lift a basic MK model toward measurement accuracy.

---

## 2. Enhanced Motley-Keenan physics (the core of v2)

The full MK form v2 targets:

```
L = L_FS(d,f) + L_c + Σ_i K_wi · L_wi(f, θ_i, t_i) + Σ_j K_fj · L_fj   (+ diffraction terms)
```

- `L_FS(d,f)` free-space loss (Friis, unchanged — the one exact law).
- `L_c` constant clutter/body term (v1's Beer-Lambert per-meter furniture
  generalizes this).
- `K_wi` number of crossings of wall type i (R3: contiguous runs count once).
- `L_wi(f, θ, t)` per-wall loss, now a **function** of frequency, incidence
  angle, and thickness — the v2 upgrade — instead of a constant.
- `Σ K_fj L_fj` floor term (FAF), scaffolded for v3/v4 vertical work.

### 2.1 Frequency-scaling exponent γ
Replace the crude global multiplier with a physical power law per material:
`L_wall(f) = L_ref · (f / f_ref)^γ`, f_ref = 2442 MHz. The Gflex's
**simultaneous multi-band logs** are how you solve for γ per material — the
same wall measured at 619, 1935, and 2600 MHz gives three points to fit the
exponent. Expect γ ≈ 0.3–0.6 for drywall/glass, higher for concrete. Until
the fit exists, seed from the measured 2.4/5 GHz attenuation table already in
`LOSS_DB_V2` (phase_a.py).

### 2.2 Angle of incidence
v1 assumes every ray hits every wall at 90°. Real corridors send rays through
walls at oblique angles, which increases the effective path length through
the material and the reflection/refraction losses. v2 scales per-crossing
loss by the ray's arrival angle relative to the wall normal (estimable from
the rasterized wall orientation, or exactly once vector tracing exists —
STEP_1 roadmap item 3). This is the enhancement that aligns predictions with
the SINR swings the Gflex records while walking corridors.

### 2.3 Wall-thickness scaling
v1 splits only thick (concrete) vs thin (drywall). v2 scales the specific
attenuation with the **rasterized run length** of each material along the ray
(the Beer-Lambert-per-meter form already used for furniture, generalized to
walls — this is spec §12.3-1). A double-thick concrete wall then costs
proportionally more without a new class.

### 2.4 Diffraction (retire the saturation stand-in)
v1's obstruction saturation (linear→40, cap 90 dB) is an admitted placeholder
for energy that really arrives by bending around corners. v2 implements a
**single knife-edge diffraction fill** (Fresnel parameter ν, loss
`J(ν)=6.9+20·log10(√((ν−0.1)²+1)+ν−0.1)`, combined with the through-wall path
in **linear power**), or the Recursive/UTD diffraction model for the atrium
and long-hallway cases. **When this lands, DELETE `saturate_obstruction` —
do not stack them** (DECISIONS.md D10).

### 2.5 Floor Attenuation Factor (FAF)
Scaffolded, not exercised on one floor: `Σ K_fj·L_fj` for signals crossing
concrete ceilings/floors. This is the hook for vertical walk-tests and the
outdoor-donor-through-multiple-floors path in v3/v4.

---

## 3. Material changes (measured, not assumed)

The most consequential v2 change and the reason a class un-fold is needed:

| class | v1 loss | v2 loss | why |
|---|---|---|---|
| exterior envelope | 3 dB (clear glass) | **20 dB** | FCC HQ facade is **low-E / metal-coated glass** — measured ground truth |
| interior / lunch-room glass | folded into exterior | **~3 dB (normal glass)** | lunch-room enclosure is ordinary glass — must be its **own class** now |

**This is a class-set change (cascade C in the retrain runbook).** In v1,
glass_partition (id 6) folds into exterior_glass (id 5) because both were
~3 dB. At 20 vs 3 dB they can no longer share a channel — v2 is a **7-class**
one-hot model (`IN_CH` grows by 1; edit MATERIALS, FOLD map, manifest, the
notebook, and simulator_tab.js — see runbook §4.9).

**Downstream consequence worth its own study:** a 20 dB low-E facade changes
the O2I (outdoor-to-indoor) penetration from the current 15 dB to ~20–28 dB,
which means outdoor cellular bleeds in *far less* than v1 assumes — directly
relevant to the co-channel interference analysis (§6) and a real argument for
where an indoor transmitter helps.

---

## 4. Frequency plan (two distinct sets, unioned)

v1 conditioned on 4 assumed indoor-Tx bands only. v2 must cover **both** the
bands the scanner actually measured (for calibration and the outdoor story)
and the hypothetical indoor-transmitter bands (for the coverage tool):

- **Measured by the Gflex (T-Mobile donors):** 619, 627, 1935, 2510, 2600 MHz
  (n71, n71, n25/n2, n41, n41).
- **Assumed indoor transmitter (the model's "what-if" bands):**
  2442 / 3500 / 5500 / 6125 MHz (Wi-Fi 2.4/5/6 GHz, NR n78).

v2 dataset conditions on the **union (9 anchors)** so the surrogate can
predict at the exact frequencies the walk test recorded (enabling per-band
calibration) *and* at the indoor deployment bands. The frequency-feature
normalization widens to span 619–6125 MHz.

---

## 5. Rasterization improvements (Indoor Walk Test v3 grid)

Tied to `STEP_1/README.md` roadmap, prioritized for v2:
1. Fix known misclassifications via `material_overrides.json` (lunch-room
   extent, elevator glass doors, aluminum cubicle spines).
2. **Separate the two glass classes** (§3) — the override file gains an
   exterior-vs-interior glass distinction.
3. Door detection (swing-arc template match) so doorways stop being sealed by
   consolidation — unlocks the angle-of-incidence and eikonal work.
4. (Stretch) vector wall tracing in QGIS — exact thickness + orientation per
   segment, which §2.2/§2.3 want, and the format the outdoor ray tracer needs.

---

## 6. Co-channel interference analysis (new research contribution)

The scanner recorded strong outdoor **2.6 GHz** (T-Mobile n41 at 2510/2600
MHz) bleeding indoors. A hypothetical indoor transmitter in the same 2.6 GHz
block (LTE B7/B38 or NR n7/n41) would **co-channel clash** with those donors.
This is exactly the problem the Combined tab and the hole-filling optimizer
already frame — v2 makes it quantitative.

- **Why they interfere:** identical-frequency waves overlap and distort
  (co-channel); the 2.6 GHz range is shared/coordinated spectrum; outdoor
  signals penetrate the facade and clash with indoor Tx.
- **What v2 adds:** with the 20 dB low-E facade (§3), the model can predict
  where outdoor n41 is strong enough indoors to matter, and the optimizer's
  hole-filling objective places the indoor Tx where the macro is weak —
  turning "avoid co-channel collision" into a map.
- **Mitigations the tool can now reason about:** (a) frequency coordination —
  put the indoor Tx on a non-overlapping block; (b) RF shielding — the low-E
  glass IS the shield, and the model quantifies how much it helps; (c) power
  control — lower indoor Tx power to minimize local collisions (the EIRP
  slider already exposes this).

---

## 7. Outdoor sandbox foundation (v3 / v4 groundwork)

Lay the foundation — do not build the outdoor model yet:
- Add a **2D OpenStreetMap basemap layer beneath the 7th-floor plan**,
  georeferenced to the same frame (the QGIS affines already put the floor at
  known lon/lat around **FCC HQ, 45 L Street NE, Washington DC**).
- Leave the surrounding blocks as empty canvas — the place where v3/v4 will
  model outdoor donor sites, street-level propagation, and the true
  outdoor→facade→indoor cascade the Gflex actually measured.
- This reframes the project from "one floor" toward "a floor embedded in its
  real RF neighborhood," which is where the outdoor 2.6 GHz interference
  story ultimately gets validated.

Concretely for v2: a non-functional OSM tile layer + the coordinate plumbing,
so the map shows the building in context. No outdoor physics in v2.

---

## 8. ARFCN / band-table integration (DONE in the dashboard)

Already shipped: the sidebar lists the **observed ARFCNs** as buttons
(channel number derived from measured frequency via the NR global raster /
LTE EARFCN offsets) with provider + band + measurement count, and clicking
one filters the map. This is the "channel number → band table" surface — it
tells you which bands the Gflex saw and how many samples each has, which is
what drives the frequency plan (§4) and the interference analysis (§6).

---

## 9. RF property coverage (v1 → v2 progression)

| RF property | v1 | v2 (planned) |
|---|---|---|
| Free-space spreading | ✅ Friis, n=2 | ✅ unchanged |
| Wall transmission loss | ✅ per-crossing dB | ✅ **frequency + angle + thickness dependent** (§2) |
| Bulk absorption (clutter) | ✅ Beer-Lambert dB/m | ✅ unchanged |
| Diffraction | ❌ saturation stand-in | ✅ **knife-edge / UTD fill** (§2.4) |
| Reflection / corridor waveguiding | ❌ | ~ partially via angle-of-incidence + calibration |
| Multipath fading / interference | ❌ (margin only) | ~ co-channel interference **mapped** (§6); fast fading still median-suppressed |
| Angle-dependent wall loss | ❌ | ✅ **§2.2** |
| Fresnel/Snell, polarization | ❌ | ❌ (needs full ray tracing — v4) |
| Delay spread | ❌ | ❌ (possible 2nd output head — v4) |
| Floor attenuation (FAF) | ❌ (1 floor) | ~ **scaffolded** (§2.5) |
| Doppler / mobility | ❌ (static) | ❌ (static maps by design) |

---

## 10. The Gflex → enhanced-MK calibration workflow (Phase D, formalized)

```
[ Gflex walk-test ]  → captured RSRP / freq / location (known Tx!)
        │
        ▼
[ Empirical calibration engine ]  → least-squares fit of:
        │                            · per-material L_ref (penetration loss)
        │                            · per-material γ (frequency exponent, §2.1)
        │                            · optional angle/thickness coefficients
        ▼
[ Refined multi-wall prediction map ]  → regenerate dataset → retrain surrogate
```

1. **Material-constant fit:** Gflex records exact RSRP at known coordinates →
   `scipy.optimize.least_squares` solves the real penetration loss L_w per
   wall type (bounded ±50% of the literature seed). `phase_d_calibrate.py`
   already does the per-class version.
2. **Frequency exponent γ:** the multi-band logs (619/1935/2600 MHz through
   the same walls) fit γ per material — the v2-specific step.
3. **Angle & FAF (stretch):** with layout-derived incidence angles and
   vertical walk data, fit the angle and floor coefficients.
4. **Hold out 20%** of measurement points as the honest field-test set;
   acceptance ≤ 8 dB RMSE (the shadow-fading floor).

**Blocking prerequisite (unchanged from v1):** this needs a walk with a
transmitter of **known power at a known position**. The archived walk data is
outdoor T-Mobile donors at unknown sites — usable for the outdoor story and
level anchoring, not for indoor per-wall calibration.

---

## 11. Implementation workflow (ordered task list)

Each maps to the retrain cascade in `retrain_for_physics_training_map_v2.md`.

1. **Rasterization** (cascade B/C): un-fold interior vs exterior glass;
   overrides cleanup; (stretch) door detection. → new grid, new manifest hash.
2. **Material table**: exterior glass 3→20 dB; add interior-glass class at
   3 dB; 7-class one-hot. → `MATERIALS`, FOLD map, manifest, IN_CH, notebook,
   simulator_tab.js (runbook §4.9).
3. **Enhanced-MK engine** in `phase_a.py`: frequency exponent γ (§2.1),
   thickness scaling (§2.3), angle of incidence (§2.2). Pull exact forms from
   IEEE 8016211. Add unit tests (γ monotonic in f; oblique > normal loss).
4. **Diffraction** (§2.4): implement knife-edge fill; **remove
   `saturate_obstruction`**. Re-run `--test`.
5. **Frequency plan**: 9-anchor union (§4); widen freq normalization.
6. **Dataset** (cascade A/C): `make dataset` with the new physics/classes;
   audit; re-check clip stats (20 dB glass will shift them).
7. **Train**: `phase_c_train_colab_v3.ipynb`, adjust IN_CH; seeds {0,1,2}.
8. **Calibrate** (Phase D) if a known-Tx walk exists: fit L_ref + γ,
   regenerate, fine-tune.
9. **Deploy**: export ONNX (single-file, >50 MB, parity ≤0.1 dB); `make
   assets`; bump `?v=N`; release `surrogate-v2` + `dataset-v2`.
10. **Outdoor foundation** (§7): OSM basemap layer + coordinate plumbing only.

---

## 12. Acceptance criteria for v2

- Enhanced-MK unit tests pass (frequency exponent, angle, thickness, R3).
- Diffraction implemented and the saturation stand-in removed.
- Interior/exterior glass are distinct classes; exterior = 20 dB.
- Surrogate beats v1's 4.68 dB test RMSE **or** the physics change is shown to
  make the map harder-but-more-real (document which; a physics improvement can
  legitimately raise sim-RMSE while lowering *field* RMSE — that's the point).
- If Phase D ran: ≤ 8 dB RMSE on held-out measured points (the number that
  actually claims agreement with reality).
- OSM basemap renders the building in its FCC-HQ context; no outdoor physics.
- Old releases kept for the physics-ladder ablation (multiwall vs enhanced-MK
  vs calibrated — field RMSE per rung).

---

## 13. Scope & honesty (carried from v1, still true)

7th-floor-specific; one output (path loss, R1); Tx power/gain applied
downstream (R2); maps combine in linear power (R7). Simulation-trained,
measurement-calibrated only after Phase D. Every material constant is
literature until the Gflex fit runs. mmWave (FR2, 24–48 GHz that the Gflex can
capture) and polarization/delay-spread remain out of scope until ray tracing
(v4).
