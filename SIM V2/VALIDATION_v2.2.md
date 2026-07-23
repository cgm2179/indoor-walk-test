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

## 3b. RESOLUTION — effective-obstruction calibration (FIXED, validated)

The over-count is **geometric**: the median indoor cell's straight ray crosses
**9 wall runs** (measured), but real energy routes through doorways/corridors
the raster lacks, crossing far fewer *solid* walls. So the fix is not a brute
cap (which flattens everything) but a two-parameter effective-path correction,
added to both engines as `engine_v2.effective_obstruction` (default no-op, so
all 37 self-tests + parity still pass):

```
obs_eff = ceiling · tanh( solidity · obs / ceiling )
```
- **solidity 0.35** — fraction of the summed solid-wall loss the dominant path
  actually incurs (the rest is doorways/routing); ~3 effective crossings of 9.
- **ceiling 55 dB** — soft "an alternate path always exists" bound; stops
  deep-shadow cells running to 300+ dB. `tanh` keeps small obstruction ~linear,
  so near-Tx structure is preserved.

Tuned to ITU-R P.1238 office plausibility, then **cross-validated over 5 random
transmitters with full diffraction** (`validate_v2_physics.py`, a re-runnable
test case). Result — indoor path loss, calibrated:

| band | p10 | median | p90 | clip |
|---|---|---|---|---|
| 619 | 54 | 77 | 102 | 0% |
| 2600 | 72 | 111 | 129 | 0% |
| 3500 | 77 | 118 | 133 | 0% |
| 6125 | 90 | 130 | 138 | 0% |

All gates pass: median within physical windows, **monotonic in frequency**,
**52 dB** frequency slope (physical ~35–55), **55 dB** dynamic range (structure
preserved, not flattened), **0% clip**, and the scanner absolute bound holds —
an indoor 23 dBm Tx gives median **−88 dBm @2600**, correctly just beating the
measured outdoor-donor **−93 dBm** (indoor source is closer than a distant
macro). Clip window tightened back to [40,170] (v1's) since p99.9 is now 141 dB.

**Note on the scanner metric:** the per-band median RSRP is *donor-dominated*,
not propagation — adjacent 617/627 MHz channels differ by 23 dB because they
are different cells, so a clean quantitative frequency slope is not extractable.
The defensible scanner check is the **absolute bound** above. Rigorous
per-material calibration still needs a known-Tx walk (Phase D); `solidity`/
`ceiling` are ITU-anchored placeholders the `jitter` field makes a fine-tune.

## Status — READY (with the honest caveat)
- Pipeline: **runnable and validated** end to end (CPU smoke green, calibrated
  targets 77–131 dB, 0% clip). Wired through `manifest_v2.json` (R6).
- Physics: **calibrated to physical plausibility**, passing the re-runnable
  cross-validation. Constants are literature/ITU-anchored, not yet fit to this
  building — Phase D refines them. Safe to generate the full dataset on GPU.
- Before the full run: `phase_a_v2 --prepare` (writes grid+manifest),
  `validate_v2_physics.py` (must print PASSED), then generate.
