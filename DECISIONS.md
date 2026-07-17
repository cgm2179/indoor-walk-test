# Architecture & Decision Record

Every load-bearing decision in this project, with the reasoning and the
evidence that forced it. Read this before changing anything structural —
several of these decisions were reached by *hitting the failure first*.
Companion docs: [README.md](README.md) (layout), [SIM/README.md](SIM/README.md)
(phase map), [SIM/MODEL_CARD.md](SIM/MODEL_CARD.md) (model scope/limits).

## The core architecture (why three layers)

```
physics generator  ->  dataset  ->  UNet surrogate  ->  browser
(slow, exact)          (frozen)     (fast, approximate)  (interactive)
```

**D1 — The neural network is a cache, not a physicist.** All physical truth
enters at the generator (equations) or Phase D (measurements). The UNet only
makes evaluation ~1000x faster so a browser can afford interactivity. This is
the FlexRDZ pattern (arXiv:2309.01861). Consequence: never tune physics by
editing the model; edit the generator, regenerate, retrain.

**D2 — The surrogate never knows which physics made its targets.** Dataset
format, model, manifest, and web app are physics-agnostic; only
`SIM/phase_a.py` changes when the physics ladder climbs (multiwall ->
diffraction -> ray tracing). This is why the phases are separate files.

**D3 — manifest.json is the single source of truth (Rule R6).** Every
constant (losses, n, d0, saturation, clip window, channel order, normalizers)
lives in `SIM/manifest.json` and nowhere else. Python and JavaScript read it;
JS/Python physics parity was verified at 0.034 dB. If you hard-code a number
in two places, you have created next month's bug.

**D4 — Datasets store only targets.** Inputs (one-hot floor + Tx blob + freq
+ distance) are rebuilt on the fly from (grid, tx, f). This is why the
9th-channel fix (D12) required **no dataset regeneration**. Keep it that way.

## Geometry / rasterization decisions

**D5 — Measured scale over estimates.** 0.0679 m/px from the 3 QGIS ground
control points in the `.TAB`. A full 6-parameter affine fit blew up (produced
a 276x309 m floor) because the GCPs are nearly collinear; a similarity fit
(uniform scale + rotation) is immune. GCP residuals 1.2-2.4 m are GPS-grade
placement error and are fine: room geometry is pixel-exact from the drawing;
residuals affect only the geo-anchor and scale by <1 dB of path loss.

**D6 — Max-loss pooling, not nearest-neighbor, for the 256x448 model grid.**
Walls are ~2 px in the source raster; NN at a 2.57x stride deletes them at
random. Any-hit pooling for walls preserves them; furniture pools by
majority so single-pixel chairs don't inflate clutter. (First attempt used
any-hit for everything: furniture ballooned and medians blew past 200 dB.)

**D7 — 6 material classes, folded from 8.** The spec fixes 6 one-hot input
channels. glass_partition (id 6) folds into exterior_glass (both ~2-3 dB);
cubicle_aluminum (id 7, zero pixels assigned) folds into furniture. Walked
ground truth from the building owner: exterior + lunch room = glass, columns
= drywall-wrapped, furniture soft/wood.

## Physics decisions (v1.1 — each reached by hitting the failure)

**D8 — n = 2.0 (free space) with explicit walls.** Using an office n≈3 AND
counting walls double-charges the environment (observed: median -134 dBm,
nonsense). Motley-Keenan convention: distance term is free-space; walls carry
the environment.

**D9 — Furniture is bulk clutter (0.3 dB/m), not per-crossing.** At 0.17 m
cells an open-plan ray crosses ~50 drawn objects; the spec's 1 dB/crossing
added +50 dB of fiction. This pre-adopts the spec's own 12.3-1 (Beer-Lambert)
for the one class where run-counting fails. Walls stay per-crossing (R3:
contiguous runs count once — unit-tested in phase_a --test).

**D10 — Obstruction saturation: linear to 40 dB, smooth cap at 90 dB.**
Straight rays over-punish deep shadow (observed: 16 drywall + 3 concrete +
4 cores = 217 dB on one diagonal; real energy arrives via diffraction and
corridors). The saturation is an explicit, documented stand-in for the
12.3-2 knife-edge rung — REMOVE IT when that rung is implemented, don't
stack them.

**D11 — Clip window [40, 170] dB (spec said [40, 150]).** This floor's
physics tops out ~170; the doc's window clipped ~50% of cells. 8.86% still
sit at the 170 ceiling — those are true dead zones (undetectable RSRP), the
model learns "saturated", and that's the correct answer there.

## Model decisions

**D12 — 9 input channels, not the spec's 8.** The 8-channel input (one-hot 6
+ tx blob + freq) stalled at 16 dB val RMSE for 12 epochs: the net had to
infer 20log10(d) from a sigma=2-cell dot. Channel 8 = log10(dist m)/3 —
geometry, not a Tx parameter, so R2 (no power/gain inputs) stands. Result:
11.4 dB after ONE epoch, 4.6 dB converged. Also: loss and metrics are masked
to indoor cells (30% of the map is outside padding).

**D13 — Tx power/gain/antenna are never model inputs (R2).** Path loss is
independent of them by definition; they're applied downstream:
P_rx = P_tx + G - PL. Maps combine in linear power (R7), never in dB.

**D14 — Shadowing X_sigma is not in training targets.** MSE regresses to the
conditional mean, so adding noise only slows convergence. It lives in the
optimizer's fade margin (z·sigma_SF) and the Phase D acceptance floor.

**D15 — Optimizer is exhaustive search, not RL.** Fast surrogate => evaluate
every walkable candidate with an explicit objective; returns top-5 (never
argmax). Transparent, reproducible; FlexRDZ's own planner-vs-PPO comparison
supports this.

## Shipping / infrastructure decisions

**D16 — Model ships as a GitHub Release asset (surrogate-v1), not in git.**
124 MB > GitHub's 100 MB limit. fp16 halving (62 MB) was tried and FAILED
the 0.1 dB parity gate (0.38 dB indoor — intrinsic float16 accumulation;
BatchNorm exclusion didn't help). Full precision or nothing. `make model`
fetches it. Dataset likewise: release `dataset-v1`, `make dataset-fetch`.

**D17 — Colab round-trip hazard.** "Save a copy in GitHub" replaces the whole
notebook file; it reverted a repo-side fix once. Rule: re-open the notebook
from GitHub before saving from Colab, and treat repo-side notebook edits as
the source of truth.

**D18 — Training must survive disconnects.** Colab kills sessions; full
training state (net + optimizer + LR schedule + epoch) checkpoints to Drive
every 2 epochs and auto-resumes. Never write a >30-min Colab loop without
this.

## Known-wrong things, ranked by fix priority

1. **Uncalibrated losses** — every dB in the table is literature, not this
   building. Phase D (known-Tx walk + least_squares fit) is THE highest-value
   next step and the riskiest (spec agrees). `SIM/phase_d_calibrate.py` is
   ready.
2. **Frequency scaling too gentle** — measured tables show concrete/glass
   ~2x from 2.4->5 GHz; v1.1 global multiplier under-scales (~10 dB low on
   concrete at 5.5 GHz). Fix is QUEUED: set `PHYSICS_VARIANT = "v1.2"` in
   phase_a.py -> `make dataset` -> retrain. Values already in `LOSS_DB_V2`.
3. **Single seed** — 4.68 dB is seed-0 only; R9 wants mean±std over {0,1,2}
   (`QUICK = False` in the notebook).
4. **No diffraction** — deep shadows are saturation-shaped, not
   physics-shaped (see D10; spec 12.3-2 is the rung).
5. **BS mode is coarse** — 8 precomputed bearings, source-averaged
   normalization (P_ref is a model-frame parameter, anchored to the walk-test
   median at +11 dBm, bearing unknown until Phase D).
