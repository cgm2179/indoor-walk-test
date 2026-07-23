# Validation report — SIM v2.2 enhanced-Motley-Keenan bundle

Reviewed the LLM-generated v2.2 bundle (`SIM V2.2 (1)/`). Two categories of
finding: **(1) the pipeline could not run as delivered** — now fixed; and
**(2) the physics, though internally excellent, over-punishes this floor at
mid/high frequency** — a genuine content issue the dataset must not ship with.

## 1. Runnability — was broken, now fixed

The `phase_b_v2_generate_colab.ipynb` notebook calls two scripts that **did
not exist anywhere** in the bundle or repo:
- `SIM V2/phase_a_v2.py` (7-class grid un-fold + `manifest_v2.json`)
- `SIM V2/phase_b_v2_generate.py` (the actual GPU dataset generator)

and none of the engine files (`physics_v2.py`, `engine_v2.py`,
`engine_v2_torch.py`) were committed to the repo the notebook clones. So a run
would clone a repo missing every file it invokes and fail at the first
subprocess.

**Fixed here:**
- engines copied to `SIM/` (where the notebook expects them);
- `phase_a_v2.py` written — re-pools `STEP_2/material_grid_consolidated.npy`
  (which retains interior glass as class 6) into a 7-class 256×448 grid,
  separating exterior low-E glass (id 5) from interior glass (id 6), and writes
  `manifest_v2.json` (clip [40,230], 9-band, `in_ch=10`, grid sha). Verified:
  classes 0–6 present, interior glass 848 cells, walkable 66,953 (= v1);
- `phase_b_v2_generate.py` written — wraps the torch engine, v1-compatible
  shards (`tx_pos/freq_feat/target/jitter/pos_id`), octant-stratified
  splits-by-position, resume, `--smoke/--audit/--shard-mod/--shard-rem`.
  **CPU smoke passed** end to end: 12 positions × 9 bands → correct shard,
  targets in [0,1], 108 samples.

The engine self-test suites all pass and are genuinely thorough:
`physics_v2 --test` (30 tests: P.2040 permittivity, Fresnel slab, low-E sheet,
UTD), `engine_v2 --test` (7 geometry), `engine_v2_torch --parity` (numpy↔torch
0.0007 dB). **This is high-quality physics code.**

## 2. Physics — over-punishes this floor at ≥2 GHz (SHOWSTOPPER for the dataset)

Generating a full dataset now would produce **physically implausible,
mostly-clipped targets** at the indoor-transmitter bands. Evidence, indoor
cells only, full 16-relay diffraction, central Tx:

| band | FSPL | obstruction | total PL | % cells at 230 clip |
|---|---|---|---|---|
| 619 MHz | 55 | **57** | 112 | 4% |
| 1935 | 65 | **115** | 180 | 41% |
| 2510 | 67 | **137** | 205 | 50% |
| 3500 | 70 | **174** | 244 | 60% |
| 5500 | 74 | **249** | 323 | 72% |
| 6125 | 75 | **273** | 348 | 75% |

A **median** indoor cell (not a deep shadow) showing **174 dB of wall loss at
3.5 GHz** and **273 dB at 6 GHz** is not physical. Three independent checks:

1. **Absolute:** real indoor path loss at 3.5 GHz over a floor is ~80–140 dB,
   not 244.
2. **vs v1:** v1's indoor median obstruction was ~40–60 dB (it capped
   obstruction at 90 dB precisely because straight-ray multi-wall
   over-punishes — DECISIONS D10). v2 is ~120 dB higher.
3. **vs the building's own data:** the scanner measured n41 (2.5 GHz) indoor
   median RSRP −111 dBm from outdoor donors. v2's 205 dB median total implies
   −140 to −155 dBm — **v2 is ~30–44 dB too lossy at 2.5 GHz** against the
   very measurements it's meant to be calibrated to.

### Root cause
v2 **removed v1's saturation cap** on the strength of adding UTD diffraction.
But on a *partition-dense* office floor, the direct ray to a median cell
crosses ~15–20 walls; with the frequency-scaled per-crossing losses (concrete
22 dB @3.5 GHz / 35 @6 GHz, core 28/40, drywall 4.8/8.9) and nothing capping
the sum, obstruction runs to 170–270 dB. UTD adds an *alternative* path, but
the diffracted legs also cross many walls, so the fill is far too weak to
rescue the bulk of cells. This is the exact over-punishment v1 diagnosed
("217 dB on one diagonal") — v2 re-exposed it by deleting the cap.

### Options (a physics-design decision — not silently changed here)
1. **Re-introduce a soft obstruction cap** (v1-style saturation, or a smooth
   soft-max), tuned so the median matches measurement. Cheapest, most robust;
   partially undoes v2's "no saturation" premise but that premise doesn't hold
   on this floor.
2. **Calibrate first (Phase D).** The per-material scales are pure literature;
   the measured data says mid-band should be ~30–50 dB lower. A least-squares
   fit would pull the constants down — but calibration fixes *scale*, not the
   structural blow-up in deep multi-wall paths, so a cap is likely still needed.
3. **Strengthen the multipath model** (more diffraction edges/reflections) so
   real low-loss paths — corridors, reflections — actually appear. Most
   physical, most expensive.
4. **Widen the clip and accept high bands are "dead."** Rejected: 50–75% of
   *indoor* cells clipped at the primary indoor-Tx bands (3.5/5.5/6 GHz)
   trains the model on almost no signal where it matters most.

**Recommendation:** option 1 (a tuned cap) to make the dataset usable now,
then option 2 when a known-Tx walk exists. Do **not** generate the full
dataset until the median indoor PL at 3.5 GHz is in a physical range
(~100–140 dB, near-zero clip).

## Status
- Pipeline: **runnable** (missing scripts written, engines placed, smoke green).
- Physics: **correct in isolation, mis-scaled for this floor** — fix the
  obstruction blow-up before generating. The generator, sharding, splits,
  audit, and phase_c consumption are all validated and ready for the corrected
  physics.
