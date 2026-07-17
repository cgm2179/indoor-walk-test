# retrain_for_physics_training_map_v2

**Purpose.** This is the complete context-and-runbook file for producing the
next generation of the path-loss surrogate ("v2") with improved physics and
improved floor-plan rasterization. There was no such file for v1 — v1's
context lived in one long working session — so this file ALSO reconstructs
that entire history: every stage, every decision, every failure and its fix,
every number. It is deliberately long and self-contained so that a future
session, collaborator, or retrieval system can load any section of it and
have enough context to act. Sections repeat key facts on purpose; treat each
heading as independently readable.

Companion documents: `DECISIONS.md` (repo root — decision records D1–D18),
`SIM/README.md` (phase map), `SIM/MODEL_CARD.md` (model scope),
`STEP_1/README.md` (rasterization roadmap). This file goes deeper and wider
than all of them, at the cost of length.

---

## PART 1 — WHAT THIS PROJECT IS

A digital twin of one office building floor (the "7th floor", a 78.1 m x
35.0 m plate in Washington DC) for indoor RF propagation. The product is an
interactive Simulator tab inside `Frontend_Data_Display.html` where a user
places a transmitter on the floor plan and sees predicted coverage
(RSRP-style received power) instantly, can animate the wavefront arrival,
and can ask an optimizer where the best transmitter location is.

The architecture is a three-layer cache hierarchy:

```
LAYER 1: physics generator (SIM/phase_a.py)   slow, exact, editable
LAYER 2: dataset (10,000 simulated maps)      frozen snapshot of layer 1
LAYER 3: UNet surrogate (pl_unet.onnx)        fast approximation of layer 2
CONSUMER: browser Simulator tab               interactive, uses layer 3,
                                              falls back to a JS mirror of
                                              layer 1 when no model exists
```

The single most important architectural fact: **the neural network contains
no physics.** It is a compression of whatever the generator produced. To
improve the physics you edit the generator and re-run layers 2 and 3. You
can never "patch" the model. This is why this retrain runbook exists.

The validated v1 result: test RMSE 4.68 dB / MAE 2.87 dB / bias −0.09 dB
against simulator ground truth on transmitter positions never seen in
training, versus baselines FSPL-only (72.8 dB), fitted log-distance
(17.6 dB), and 3GPP TR 38.901 InH-Office with exact ray-marched LOS
(58.9 dB). The floor-plan-aware surrogate beating the strongest map-blind
standard by ~54 dB RMSE is the project's central demonstrable claim.

---

## PART 2 — FULL PROJECT CHRONICLE (the v1 context that was never written down)

### 2.1 Origins: the walk test

The project began with a physical walk test on the 7th floor using a PCTEL
SeeHawk G-flex scanner. Raw exports live in `ARCHIVE/raw_walk_data/` (CSV/,
DTR/, Device032409007/, concatenated into
`Concat_Indoor_Walk_Test_from_csv.csv`). Contents: 23,017 rows, of which
10,248 carry a power reading — 5,951 LTE rows ("Ref Signal - Received
Power") and 4,297 NR rows ("SSB - Received Power"), from T-Mobile macro
donors (MCC 310 / MNC 260), bands B71/617 MHz, B2/1960, n41/2506, n77/~3750.
GPS drifts badly indoors: only 2,409 of 10,248 points land on the floor
plate (the rest scatter over ~340 m). Measured indoor medians: LTE RSRP
−99.0 dBm, NR SSB RSRP −111.3 dBm. CRITICAL LIMITATION: these measurements
are from OUTDOOR macros at UNKNOWN locations, so they cannot calibrate the
indoor wall-loss model directly (you can't compute predicted loss without
knowing where the transmitter is). They ARE usable as a level anchor (see
P_ref, section 2.9) and for the eventual outdoor-to-indoor fit.

### 2.2 STEP_1: rasterization of the floor plan (v1 of the material grid)

Input: `7th_Floor_2nd_Indoor_Walk_Test_V2.2.png` (1150x515 CAD render) +
`.TAB` MapInfo georeference with 3 ground control points (pixel <-> lon/lat).

Scale: a full 6-parameter affine fit across the 3 GCPs EXPLODED (produced a
276x309 m floor) because the GCPs are nearly collinear (cross product ~5442
vs ~413k for well-spread points) — ill-conditioned perpendicular to their
line. Fix: similarity transform (uniform scale + rotation + translation +
y-flip), immune to collinearity. Result: **0.0679 m/px**, floor plate 78.1 x
35.0 m, GCP residuals 1.19/1.25/2.44 m (GPS-grade placement error;
acceptable because room-to-room geometry is pixel-exact from the drawing —
residuals only move the geo-anchor and scale by <1 dB of path-loss effect).
To tighten: add a 4th GCP in QGIS placed OFF the line of the current three.

Classification: classical CV on numpy/scipy (no OpenCV — not installed on
the Mac; scipy.ndimage does everything). Pipeline: grayscale threshold
(dark < 120 = structure) -> connected-component size separates walls
(networks of thousands of px) from room-label text (small blobs) — chosen
over morphological opening because interior walls are only ~2 px wide and a
3x3 opening deletes them -> distance-transform thickness splits concrete
(radius >= 2 px ~ >=0.3 m) from drywall -> local dark-density (15x15 window
> 28%) finds hatched cores (elevators/stairs/WC), then binary closing makes
cores SOLID (critical for physics, see 2.4) -> flood fill from image borders
finds the outside, structure within 7 px of outside = exterior envelope ->
light-gray band 150-240 = furniture. Green pin markers (two candidate Tx
locations) are extracted at subpixel precision — (636.3, 403.3) and
(663.0, 403.5) px, 1.8 m apart — then erased before classification.

Ground truth from walking the floor (owner's knowledge, encoded in the
loss table): exterior envelope is GLASS curtain wall (3 dB, not 15);
the curved lunch-room enclosure at floor center is GLASS; columns are
DRYWALL-wrapped (4 dB, not concrete 15); furniture is soft/wood; cubicle
partitions are aluminum-skinned. Because glass and drywall are identical
thin lines in a CAD drawing, `STEP_1/material_overrides.json` exists:
hand-drawn rect/polygon regions re-label listed material ids after
auto-classification (rough boxes are safe — air/furniture pass through).
Currently seeded with the lunch-room box [515,185,635,320] px.

8 classes: 0 air, 1 drywall, 2 concrete/masonry, 3 core service area,
4 furniture (soft/wood), 5 exterior glass curtain wall, 6 glass partition
(overrides only), 7 cubicle aluminum panel (overrides only, zero px used).

### 2.3 STEP_2: first physics engine and the four physics bugs

`STEP_2/motley_keenan.py` — full-resolution (515x1150) Motley-Keenan. Four
bugs were found by validation, each is a permanent lesson:

1. **Hatched cores counted per hatch stroke.** Rays crossing an elevator
   shaft crossed ~10 separate 20 dB "walls" = +200 dB. Fix: cores are solid
   regions (STEP_1 closing), one crossing per shaft.
2. **Furniture counted per drawn object.** ~50 chair/desk symbols on an
   open-plan ray at 1 dB each = +50 dB of fiction. Fix: furniture is bulk
   clutter, dB-per-METER (0.3 dB/m), not per crossing.
3. **n=3 AND explicit walls double-charged the environment** (median Prx
   −134 dBm, nonsense). Motley-Keenan convention: distance term uses free
   space n=2.0; the walls ARE the environment.
4. **Straight rays over-punish deep shadow.** One diagonal legitimately
   crossed 16 drywall + 3 concrete + 4 cores = 217 dB of obstruction; real
   energy arrives via diffraction/corridors and measured indoor PL saturates
   ~110-140 dB. Fix: total obstruction loss saturates (STEP_2 used a smooth
   cap at 60 dB; SIM later refined to linear-to-40 + smooth-to-90).

Also: cells outside the building envelope are masked from stats/rendering
(their rays cross everything and mean nothing). Sanity numbers after fixes:
median indoor Prx −93 dBm, 64% of floor >= −100 dBm from one 23 dBm cell at
3.5 GHz — realistic for indoor NR.

`STEP_2/multiband_metrics.py` adds the metric layer: RSRP/RSRQ/SINR/RSSI per
band (B71/B2/n41/n77) with the two candidate transmitters modeled co-channel
at full load (the non-serving cell is the interference that shapes SINR:
>30 dB next to a Tx, ~0 dB on the equal-power line; RSRQ ceiling −10.8 dB
full-load). Formulas: n_RB = bw*0.9/(12*scs); EIRP_RE = EIRP −
10log10(12 n_RB); noise_RE = −174 + 10log10(scs_Hz) + NF(7 dB); RSSI =
10log10(12 n_RB) + 10log10(sum_lin + noise_lin); RSRQ = 10log10(n_RB) +
RSRP − RSSI. Also a simulated 145 m corridor-loop walk (CSV + figure + GIF).
KEY INSIGHT preserved into the SIM design: all metrics are cheap arithmetic
ON TOP of path loss — the ML model only ever needs to predict PL (Rule R1).

### 2.4 MATLAB bundle

`ARCHIVE/MATLAB/indoor_walk_test.mat` packages everything in shared frames
(float px / local ENU meters / lon-lat / EPSG:3857): grids, losses, affines,
Tx pins, sim maps, and all 10,248 normalized walk rows (protocol flag
1=LTE/2=NR, on_floor flag). `load_indoor_walk_test.m` plots it;
`motley_keenan.m` is a native MATLAB port. Note EPSG:3857 lengths are
inflated by 1/cos(lat) ≈ 1.285 — never measure distances in it; use the
local-meter affine.

### 2.5 The first surrogate (superseded, ARCHIVE/superseded/STEP_4 + WEB)

Before the formal spec arrived, a smaller pipeline was built: 2x-downsampled
grid (257x575 -> crop 256x568), 3-channel input (wall-loss map, clutter map,
log-distance), ~2M-param U-Net, 400 samples, trained on Colab to val MAE
5.12 dB, deployed in a standalone `WEB/index.html` with onnxruntime-web.
Superseded by the SIM pipeline but its lessons carried forward: the
log-distance input channel (see 2.7 — its removal caused the 16 dB stall),
Drive-checkpointing, ONNX-in-browser mechanics, and the drag-the-Tx UX.

### 2.6 The build spec and the SIM pipeline (phases A–F)

A formal instruction document ("UNet Path-Loss Surrogate — Design Context &
Build Instructions v1.0", modeled on FlexRDZ arXiv:2309.01861) defined
non-negotiable rules R1–R10 and phases A–F. The implementation lives in
`SIM/`:

- **Phase A** `phase_a.py`: grid prep (fold 8->6 classes: 6->5 glass,
  7->4 furniture; max-loss pooling to 256x448 with 0.1744 m square cells —
  the floor occupies rows 32–224, padding is outside; walls pool by any-hit
  so 2px walls survive the 2.57x stride, furniture by majority so chairs
  don't inflate clutter; class-3 fragments < 40 cells downgraded to drywall
  because they were rasterizer junction misfires charging 20 dB each),
  multiwall engine (R3 contiguous-run counting, unit-tested; per-meter
  furniture; obstruction linear to 40 dB then smooth saturation capped at
  90 dB — the documented stand-in for knife-edge diffraction), outdoor-BS
  plane-wave mode (illuminated facade cells re-radiate P_ref − O2I inward,
  linear-power combined and source-count-averaged — which makes P_ref a
  MODEL-FRAME parameter, not literal field strength), arrival-time maps
  (T = d/c indoor; plane-wave staggered for BS), and the manifest writer.
  0.49 s per 256x448 map on the Mac (target was <= 2 s).
- **Phase B** `phase_b_dataset.py`: 2,500 Tx positions uniform on the
  walkable mask (min spacing 2 cells) x 4 freqs (2442/3500/5500/6125 MHz) =
  10,000 target maps; 50% get per-material wall-loss jitter U(0.8,1.2);
  targets clipped [40,170] dB and normalized /130; splits BY POSITION
  2000/250/250 stratified by floor octant, seed 0, committed to
  splits.json; shards of 500 (float16); resumable and parallelizable
  (--shard-mod/--shard-rem; 4 workers ≈ 25 min on the Mac's 10 cores).
  Audit: splits disjoint, all freqs of a position together, histogram
  smooth, 0.00% at low clip, 8.86% at high clip (true dead zones —
  documented deviation from the doc's <1% hope).
- **Phase C** `phase_c_train_colab_v3.ipynb`: 9-channel input (see 2.7),
  UNet 4-down base 64 (~31M params), MSE + 0.1*gradient-L1 loss masked to
  indoor cells, AMP, Adam 1e-3 cosine->1e-5, batch 16, <=150 epochs, early
  stop patience 15, Drive-resumable every 2 epochs, seeds {0,1,2} behind
  QUICK flag, baselines (a) FSPL (b) log-distance with n fitted on train
  (c) 38.901 InH dual-state with the LOS indicator computed EXACTLY by the
  Phase A ray-march (zero crossings => LOS), test set touched once, ONNX
  export with a 0.1 dB parity gate.
- **Phase D** `phase_d_calibrate.py`: least-squares fit of {n, per-material
  loss scales bounded ±50%} to a known-Tx measurement walk. BLOCKED until
  such a walk exists (see 2.1 — current data is from unknown outdoor
  donors). D.3 physical sanity checks run today and pass.
- **Phase E** `phase_e_optimizer.py` + in-browser: exhaustive search over
  walkable cells, objectives = coverage with fade margin (P_rx − z*σ_SF >=
  threshold; z = 0.84/1.28/1.65 for 80/90/95%; σ_SF = 3.0 LOS / 8.03 NLOS
  from 38.901 until Phase D residuals replace them), mean path loss, and
  hole-filling (cover only where the BS map is weak). Returns top-5, never
  argmax.
- **Phase F** `export_web_assets.py` -> `web/sim_assets.js` (manifest +
  grid + masks + 8-bearing BS maps, base64 in a script tag so file:// works)
  and the Simulator tab inside `Frontend_Data_Display.html` (three sub-tabs
  Received/Transmitter/Combined; R10 validation — no silent defaults, Solve
  disabled until inputs set; R8 walkable snapping; R7 linear-power
  combining; static ⇄ time-lapse with leading-edge band and the on-screen
  x10^-8 time-scale label; PNG/CSV export with metadata; device presets;
  legend generated from the manifest). JS physics mirrors Python at
  0.034 dB max deviation (verified cell-by-cell).

### 2.7 The training saga (what to expect when retraining)

Chronology of failures and fixes — each is a trap a v2 retrain could hit:

1. **Dataset not in the clone**: shards are gitignored; the notebook's first
   dataset cell checked for splits.json (which IS committed) and crashed
   with "need at least one array to concatenate". Fixed: check for
   shard_*.npz, auto-search the top two levels of Drive (the user's upload
   landed at MyDrive/SIM/indoor-walk-test-dataset, not the documented path).
2. **The 16 dB stall**: the spec's 8-channel input (one-hot 6 + tx Gaussian
   sigma=2 + freq) forces the net to infer 20log10(d) from a 2-cell dot.
   12 epochs at 7.5 min each plateaued at 16.04 dB val RMSE. Fix (v3
   notebook): channel 8 = log10(max(d_m,1))/3.0 (geometry, not a Tx
   parameter — R2 intact) + loss/metrics masked to indoor cells. Result:
   11.4 dB after ONE epoch, 9.47 by ep 3, converged 4.64 dB val / 4.68 test.
3. **Colab GPU quota**: free tier cut the first run off. Mitigation baked
   in: full training state (net+opt+sched+scaler+epoch+best) checkpoints to
   Drive every 2 epochs (~370 MB rolling file), auto-resumes on rerun,
   deletes itself on clean early-stop. Epochs were 7.5 min on a T4 and
   65–73 s on the faster GPU the user later got.
4. **Colab save clobber**: "Save a copy in GitHub" replaces the WHOLE
   notebook file with the session's copy — it reverted repo-side fixes
   twice. Rule: re-open the notebook from GitHub before saving from Colab;
   repo-side edits are the source of truth.
5. **Missing packages on Colab**: newer torch needs `onnxscript` for its
   exporter and Colab stopped preinstalling `onnxruntime`. The export cell
   pip-installs `onnx onnxscript onnxruntime` first.
6. **The two-file export trap**: new torch's dynamo exporter writes
   pl_unet.onnx (16 KB graph) + pl_unet.onnx.data (124 MB weights). Only
   the graph downloaded; browser ORT failed with "Failed to load external
   data file". Python-side parity passed anyway (it could see the .data
   next to it) — deceptive! Fix: export with dynamo=False (legacy single
   file) or onnx.save(..., save_as_external_data=False), then ASSERT the
   file is > 50 MB before shipping. This gate is in the notebook now.
7. **fp16 shrinking fails parity**: converting to float16 (would be 62 MB,
   committable) produced 0.38 dB indoor / 2.26 dB max deviation vs the
   0.1 dB gate — intrinsic fp16 accumulation through ~20 conv layers;
   excluding BatchNorm didn't help. Decision: full precision only.
8. **Distribution**: 124 MB > GitHub's 100 MB in-repo limit, so the model
   ships as release asset `surrogate-v1` (`make model` fetches it); the
   1.2 GB dataset as release `dataset-v1` (`make dataset-fetch`). Drive
   holds backups (MyDrive/SIM/checkpoints/).
9. **Browser cache staleness**: python http.server + Chrome heuristics
   served stale sim_assets.js after updates; script URLs now carry ?v=N
   cache-busting queries — BUMP N whenever sim_assets.js or
   simulator_tab.js changes. VS Code Live Preview (workspace-recommended,
   port 3210, no-cache + reload-on-save) is the preferred dev loop.

### 2.8 v1 final numbers (the baseline every v2 must beat)

| model | TEST RMSE | TEST MAE | bias |
|---|---|---|---|
| FSPL only | 72.81 dB | 68.31 | −68.31 |
| log-distance (fitted n) | 17.56 dB | 13.02 | +0.80 |
| 3GPP 38.901 InH (exact LOS) | 58.91 dB | 54.96 | −54.89 |
| UNet surrogate v1 (seed 0) | **4.68 dB** | **2.87 dB** | **−0.09** |

D.3 sanity: monotone LOS decay fraction 1.0; 0.00% of cells below FSPL;
2.4 GHz <= current-band cell-wise median = 1. ONNX parity 0.0011 dB.
Formal C.5 target was <=3 dB RMSE — v1 is at 4.68 on a single seed; the
lever is more data (D-B below), not more epochs (the curve had flattened).

### 2.9 Demo presets and the outdoor P_ref anchor

The Simulator tab has a Device preset dropdown (manifest-driven): Home Wi-Fi
router 2442/20 dBm/2 dBi; Enterprise ceiling AP 5500/17/4; Wi-Fi 6E AP
6125/18/3; 5G small cell n78 3500/24/5; Phone hotspot 2442/15/0. Choosing a
preset is an explicit action so R10 (no silent defaults) holds. The
"Outdoor macro (walk-test calibrated)" preset fills P_ref = **+11 dBm** —
derived, not guessed: the value that makes the simulated BS map's indoor
median equal the measured walk-test median (−111.3 dBm over 843 on-floor NR
points); per-bearing values ranged −1.5 to +20.7 dBm (bearing unknown until
Phase D; 135° is a labeled placeholder). Remember P_ref is model-frame (the
facade-source averaging shifts it ~20 dB from literal field strength).

---

## PART 3 — CURRENT STATE SNAPSHOT (as of this file's creation)

- Repo: github.com/cgm2179/indoor-walk-test (PRIVATE). Root layout: SIM/
  (active), STEP_1/ STEP_2/ (upstream inputs), Frontend_Data_Display.html +
  timeseries_data.js (dashboard), floor-plan source files, Makefile,
  README.md, DECISIONS.md, ARCHIVE/ (raw data, early analysis, superseded
  WEB/STEP_3/STEP_4, MATLAB bundle).
- Manifest: `SIM/manifest.json` version **sim-v1.1**, 9 channels, clip
  [40,170], cell 0.1744 m, grid 256x448, floor rows 32–224.
- Physics variant: v1.1 active; **v1.2 QUEUED** in phase_a.py behind
  `PHYSICS_VARIANT` (see Part 5).
- Releases: `surrogate-v1` (pl_unet.onnx, 124,150,062 bytes, fp32, opset 17,
  input name 'x', output 'pl_norm'); `dataset-v1` (20 shards + splits +
  meta, 22 assets).
- Model deployed locally at SIM/web/pl_unet.onnx; engine badge shows
  "UNet surrogate (wasm)" (webgpu where available).
- Drive: MyDrive/SIM/indoor-walk-test-dataset (dataset),
  MyDrive/SIM/checkpoints/ (unet9_seed0_best.pt, pl_unet.onnx).
- Remaining v1 debt: seeds {1,2} not run (R9); Phase D blocked on known-Tx
  walk; physics v1.2 not applied; diffraction not implemented.

---

## PART 4 — THE RETRAIN RUNBOOK (generic, any physics/rasterization change)

The invariant: **grid + physics + dataset + model + web assets must move
together.** The manifest's grid/walkable SHA-256 prefixes are the tripwire —
Phase C records them; if they mismatch the dataset you're training on, stop.

### 4.0 Decide which cascade you're in

- **(A) loss values / physics equations changed, same 6 classes, same
  grid** -> steps 4.2–4.8 (skip 4.1). Examples: PHYSICS_VARIANT v1.2,
  Phase D calibrated losses, diffraction rung, saturation retuning.
- **(B) rasterization changed, same classes** -> steps 4.1–4.8. Examples:
  overrides edits, door openings preserved, 4th GCP rescale, PARAMS retune.
- **(C) class set changed** (new door class, low-E split) -> everything in
  (B) PLUS code edits in five places, see 4.9.

### 4.1 Rasterization refresh (cascade B/C only)

```bash
python3 STEP_1/rasterize_floorplan.py 7th_Floor_2nd_Indoor_Walk_Test_V2.2.png --out STEP_1
# inspect STEP_1/preview_materials.png — walls where walls are, cores solid,
# overrides applied (line prints px count per override)
python3 STEP_2/motley_keenan.py        # refreshes material_grid_consolidated
```
Gotchas: PARAMS in rasterize_floorplan.py are tuned to THIS 1150x515 render;
any re-export at different DPI needs re-tuning (dark_thresh 120,
min_struct_area 50, hatch_density 0.28...). The consolidated grid (STEP_2)
is what Phase A reads — don't skip it.

### 4.2 Physics edit + unit tests

Edit `SIM/phase_a.py` (for v1.2: set `PHYSICS_VARIANT = "v1.2"`). Then:
```bash
make prepare       # regenerates grid_model.npy, masks, manifest (new hashes)
make test          # A.4 crossing-count tests + D.3 sanity checks MUST pass
make sample        # visual QA: SIM/preview/*.png — shadows/corridors sane?
```
Bump `version=` in write_manifest (e.g. "sim-v1.2") so artifacts are
traceable. If the JS fallback engine's math changed (v1.2 needs the
per-material loss table branch mirrored in simulator_tab.js — the current JS
reads the single `freq_loss_mult`), port it and re-verify parity in the
browser console against `python3` probe values (v1 method: compare 4 probe
cells; accept <= 0.05 dB).

### 4.3 Dataset regeneration (~25 min on 4 cores)

```bash
rm -f SIM/dataset/shard_*.npz SIM/dataset/splits.json SIM/dataset/dataset_meta.json
for r in 0 1 2 3; do python3 SIM/phase_b_dataset.py --shard-mod 4 --shard-rem $r & done
wait
python3 SIM/phase_b_dataset.py --audit    # disjoint splits, clip % sane
```
NOTE: deleting splits.json is correct ONLY because the grid changed (position
sampling depends on the walkable mask). For a pure loss-value change
(cascade A) the walkable mask is unchanged and you may keep splits.json —
but regenerating with the same seed reproduces it identically anyway.
Deviation ledger: expect the high-clip fraction to CHANGE with new physics
(v1.1 was 8.86%); record it in the release notes. If v1.2's stronger
high-band losses push clipping way up at 6 GHz, consider widening the clip
ceiling in write_manifest (then it flows everywhere automatically — but
targets, model output scaling, and JS un-normalization all shift together,
which is exactly why it lives in the manifest).

### 4.4 Upload to Drive + train

Upload SIM/dataset/ to Drive (any top-2-level folder; the notebook
auto-finds shard_*.npz). Open
`SIM/phase_c_train_colab_v3.ipynb` FRESH from GitHub (clobber rule!), GPU
runtime, Run all. Delete any stale `unet9_seed*_resume.pt` in
MyDrive/SIM/checkpoints first (or it will resume the OLD run — the resume
file self-deletes only on clean early-stop). QUICK=True first for one seed;
QUICK=False for the R9 3-seed protocol when the result looks right. Watch:
epoch 1 should already be ~11 dB; if it stalls >15 dB for 3 epochs something
regressed (check the distance channel and mask made it into the session).

### 4.5 Evaluate against the v1 bar

The notebook prints VAL/TEST tables. v2 must beat v1's 4.68 dB test RMSE
(else the physics change hurt learnability — investigate before shipping)
AND the baselines by a wide margin. Run the D.3 sanity cell. For the
proposal, the interesting comparison is v1-model-vs-v2-physics ground truth:
that gap estimates how much the physics change actually moved predictions.

### 4.6 Export with the gates

Run the export cell (it pip-installs onnx/onnxscript/onnxruntime, forces a
SINGLE file via dynamo=False or re-embedding, asserts size > 50 MB, asserts
parity <= 0.1 dB, copies to Drive, downloads). Both asserts exist because
both failure modes actually happened (Part 2.7 items 6–7).

### 4.7 Deploy + web assets

```bash
mv ~/Downloads/pl_unet.onnx SIM/web/pl_unet.onnx
make assets            # ~10 min: re-precomputes the 8-bearing BS maps with
                       # the new physics/grid and re-embeds the new manifest
# bump the ?v=N cache-buster on the two SIM script tags in
# Frontend_Data_Display.html
```
Open via VS Code Live Preview (auto-fresh) or hard-refresh Chrome. Verify:
engine badge "UNet surrogate", a solve looks physical, hover readouts sane,
and (for physics changes) the JS-fallback and surrogate maps agree to a few
dB (temporarily rename the .onnx to force physics mode for the comparison).

### 4.8 Version + publish

```bash
gh release create surrogate-v2 SIM/web/pl_unet.onnx --title "..." \
  --notes "metrics, manifest version, dataset release, physics variant"
gh release create dataset-v2 SIM/dataset/shard_*.npz SIM/dataset/splits.json \
  SIM/dataset/dataset_meta.json --title "..." --notes "..."
# update Makefile model/dataset-fetch targets to the new tags (or keep both)
# update MODEL_CARD.md error bounds + README status line + this file
git add -A && git commit && git push
```
KEEP the old releases — the physics-ladder ablation (train the same UNet on
each physics rung, report which rung the real measurements reward) is a
reviewer-friendly result that only exists if old artifacts survive.

### 4.9 Cascade C extras (new material class)

Five synchronized edits, then the full chain 4.1–4.8:
1. `SIM/phase_a.py`: MATERIALS6 -> MATERIALS7 (+ color, losses), FOLD map,
   loss vectors sized 7, manifest channels list, IN_CH implication.
2. `SIM/manifest.json` via write_manifest: channels become
   onehot x7 + tx + freq + dist = 12? No — one-hot count = class count;
   channel indices for tx/freq/dist shift by +1. Update comments.
3. Notebook: IN_CH constant, build_input one-hot range.
4. `SIM/web/simulator_tab.js`: onehotCache loop bound, channel offsets in
   pathlossOnnx, legend renders automatically from manifest.
5. `SIM/phase_e_optimizer.py`: pl_from_onnx channel building.
Grep for "onehot", "IN_CH", "9 *" to find every site. The old model is
incompatible with the new input — never mix.

---

## PART 5 — PHYSICS V1.2, THE ALREADY-QUEUED FIRST TARGET

Measured attenuation tables (user-contributed) show per-material frequency
scaling much steeper than v1.1's global multiplier (x1.00/1.15/1.30/1.40 at
2442/3500/5500/6125 MHz). Concrete and glass roughly DOUBLE from 2.4 to
5 GHz; v1.1 is ~10 dB low on concrete at 5.5 GHz. The v2 table is already in
`SIM/phase_a.py` as `LOSS_DB_V2` (per-material absolute dB per band,
log-frequency interpolated from the 2.4/5 GHz anchors):

| material | 2442 | 3500 | 5500 | 6125 MHz |
|---|---|---|---|---|
| drywall | 3.0 | 5.3 | 8.0 | 8.7 |
| concrete | 15.0 | 22.7 | 31.5 | 34.1 |
| core (concrete+metal) | 22.0 | 28.7 | 36.5 | 38.9 |
| exterior/std glass | 3.0 | 4.5 | 6.3 | 6.8 |
| furniture dB/m | 0.30 | 0.45 | 0.62 | 0.68 |

Activation = `PHYSICS_VARIANT = "v1.2"` + runbook Part 4 cascade A. ALSO
REQUIRED: mirror the per-material table lookup in simulator_tab.js's
`pathlossPhysics` (currently reads the single global multiplier) and in the
metric layer if per-band tables matter there. IMPORTANT CAVEAT in the code:
the table assumes STANDARD glass — if the facade is low-E/coated (likely for
a modern DC office; unconfirmed), class 5 should be ~10/12/15/16 dB, a
1-line change with large consequences for the Received tab. Confirm the
glazing before shipping v1.2 as truth.

---

## PART 6 — IMPROVEMENT BACKLOG BEYOND V1.2 (ranked)

1. **Phase D known-Tx walk + calibration** — the ONLY step that converts
   literature dB into this building's dB; everything else is refinement of
   an uncalibrated model. Protocol: transmitter with known EIRP at a green
   pin, >=200 points on a known route (waypoint-click, not GPS), median per
   cell; `SIM/phase_d_calibrate.py --fit` writes calibration.json; then
   cascade A retrain (fine-tune 20 epochs per spec D.2.3). Acceptance:
   RMSE <= 8 dB on the 20% holdout (the physical floor set by shadow
   fading). Schedule this the moment Tx hardware exists — it's also the
   riskiest item (spec §9 row 1).
2. **R9 seeds {0,1,2}** — QUICK=False, two more resume-protected runs;
   report 4.7 ± x dB. Cheap, required for any formal write-up.
3. **More training data** — the 4.68->3 dB gap is a data problem (curve
   flattened at 8k train samples). `phase_b_dataset.py` N_POSITIONS=5000+
   costs ~50 min locally; diminishing returns unknown — measure.
4. **Knife-edge diffraction (spec 12.3-2)** — replaces the obstruction
   saturation with physics: for NLOS cells find the dominant convex corner,
   Fresnel parameter v = h*sqrt(2(d1+d2)/(lambda d1 d2)), loss J(v) = 6.9 +
   20log10(sqrt((v-0.1)^2+1)+v-0.1), combine diffracted path with
   through-wall path in linear power. REMOVE saturate_obstruction when this
   lands (do not stack). Then cascade A.
5. **Rasterization roadmap** — full detail in STEP_1/README.md: overrides
   cleanup (hours) -> door detection from CAD swing arcs (biggest physics
   gap; doors currently sealed by consolidation) -> vector wall tracing in
   QGIS (exact geometry, feeds Sionna) -> per-segment material learning
   from measurements -> CubiCasa5k generalization (only when a second
   building appears).
6. **Eikonal time-lapse (F.5 v2)** — |grad T| = 1/v(p) via fast-marching so
   wavefronts bend around the core and through doorways instead of Euclidean
   circles. Pure visualization; no retrain.
7. **BS mode refinement** — more bearings (15° steps), Fresnel-angle-
   dependent O2I, and after Phase D a fitted (P_ref, theta). Consider
   per-band BS precompute (currently 3.5 GHz only, other bands borrow it).
8. **Per-band surrogates or finer freq conditioning** — v1 conditions on a
   scalar freq channel across 4 anchors; if per-band accuracy matters,
   either train per-band models (the app already keys presets by band) or
   densify frequency sampling in Phase B.
9. **Delay spread / second output head** — RMS delay spread as an extra
   output for ISI analysis (out of scope for R1 today; would need R1
   renegotiation and multi-path physics — pairs with the Sionna rung).
10. **Ship the ablation figure** — same UNet trained on multiwall vs
    +diffraction vs ray-traced targets; report field RMSE per rung against
    the Phase D measurements. This isolates how much physics fidelity
    reality rewards — a publishable result whichever way it lands.

---

## PART 7 — QUICK-REFERENCE CONSTANTS AND FORMULAS (v1.1)

- FSPL(d0=1 m, f) = 32.44 + 20log10(f_MHz) − 60 dB
  (40.19 / 43.32 / 47.25 / 48.18 dB at the four bands)
- PL = FSPL(1m,f) + 10*2.0*log10(max(d,1m)) + saturate(sum walls + clutter)
- saturate(x) = x for x<=40; 40 + 50*(1−exp(−(x−40)/50)) above; cap 90 dB
- v1.1 losses (dB/crossing at 2442, x1.15/1.30/1.40 at higher bands):
  drywall 4, concrete 15, core 20, exterior glass 3; furniture 0.3 dB/m
- Normalization: target = (clip(PL,40,170) − 40)/130;
  freq_feat = (log10 f − log10 2400)/(log10 6125 − log10 2400);
  tx blob sigma = 2 cells; dist channel = log10(max(d_m,1))/3
- Grid: 256x448, cell 0.17438 m, floor rows [32,224), walkable = classes
  {0,4} AND inside; source raster 515x1150 at 0.0679 m/px
- Channels 0–8: onehot air/drywall/concrete/core/furniture/exterior,
  tx_gaussian, freq_feature, log10_distance
- Downstream math: P_rx = P_tx + G_tx + G_rx − PL (R2); combining
  P_tot = 10log10(sum 10^(Pi/10)) (R7); coverage counts
  P_rx − z*σ_SF >= threshold (z: 0.84/1.28/1.65; σ_SF 3.0/8.03 dB)
- Metric layer (from STEP_2, reusable): n_RB = floor(bw_kHz*0.9/(12*scs));
  EIRP_RE = EIRP − 10log10(12 n_RB); noise_RE = −174 + 10log10(scs_Hz) + 7
- Timing: physics map 0.49 s (Mac) / ~2 s (Colab CPU); JS physics 2–4 s;
  ONNX browser solve ms–seconds; dataset 25 min (4 workers); training
  ~1–1.5 h (fast GPU) or ~8 h (T4); BS assets ~10 min; full v2 loop ~2–3 h.

## PART 8 — ERROR -> FIX LOOKUP TABLE (everything that broke in v1)

| symptom | cause | fix |
|---|---|---|
| floor scales to 276x309 m | affine fit on collinear GCPs | similarity transform (phase_a georef; STEP_1) |
| median PL 200+ dB | furniture per-crossing + any-hit pooling + core fragments | per-meter clutter; majority pooling; core cleanup; saturation |
| median Prx −134 dBm | n=3 AND explicit walls | n=2.0 free space |
| "need at least one array to concatenate" | shards gitignored, only splits.json in clone | notebook auto-finds shards on Drive |
| val RMSE stuck ~16 dB | no distance input channel | 9th channel log10(d)/3 + indoor-masked loss |
| ModuleNotFoundError onnxscript / onnxruntime | Colab torch needs them, not preinstalled | pip install onnx onnxscript onnxruntime in export cell |
| RuntimeError tensors on two devices after failed export | net stranded on CPU by mid-cell crash | rerun export cell (ends with net.to(dev)) |
| browser: "Failed to load external data pl_unet.onnx.data" | dynamo two-file export, only 16 KB graph shipped | dynamo=False / re-embed; assert size > 50 MB |
| fp16 model 0.38 dB off | float16 accumulation through UNet | ship fp32; use release assets, not git |
| engine badge stuck on "physics" | model missing/split/stale cache | console now logs the exact ORT error; check file size; hard refresh; bump ?v=N |
| Colab save reverted repo fixes | whole-file "Save a copy in GitHub" | reopen from GitHub before saving; repo is source of truth |
| resume restarts old training | stale unet9_seed*_resume.pt on Drive | delete resume file when starting a NEW experiment |
| push rejected (fetch first) | Colab saved to main meanwhile | git pull --rebase, then push |
| preview port in use | user's own http.server on 8432 | Live Preview on 3210 / preview on 8433 |

*End of file. When v2 ships, append its chronicle here and start
retrain_for_physics_training_map_v3.*
