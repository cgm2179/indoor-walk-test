# Model Card — 7th-Floor Path-Loss Surrogate (sim-v1.0)

## Scope (read this first)

- **7th floor only.** Trained on one floor plan; predictions for any other
  geometry are meaningless. Generalizing requires training across many plans
  (CubiCasa5k-style) — future work, per spec §1.3.6.
- **One output**: path loss in dB (R1). Received power, RSRQ-style metrics,
  coverage — all derived downstream via `P_rx = P_tx + G_tx + G_rx − PL` (R2).
- **Frequencies**: 2,442 / 3,500 / 5,500 / 6,125 MHz (frequency is a model
  input; the four anchors are what the dataset covers).
- **Uncalibrated (v1)**: material losses are table values scaled by a crude
  monotone frequency multiplier; Phase D calibration against G-flex
  measurements has NOT run yet (blocked on a walk with a known transmitter).

## Physics that generated the training targets

Multi-wall Motley-Keenan, n = 2.0, d0 = 1 m, per-crossing wall losses
(contiguous runs count once, R3), furniture as 0.3 dB/m bulk clutter
(Beer-Lambert form, spec 12.3-1 pulled forward), total obstruction loss
linear to 40 dB then smoothly saturated at 90 dB (stand-in for the 12.3-2
knife-edge diffraction rung; remove when implemented). No reflections, no
true diffraction, no floor/ceiling paths. AoA/AoD, delay spread, Doppler:
out of scope (see spec §12.2).

## Known deviations from the build doc (all in manifest.json)

1. Grid is 256×448 with the floor occupying rows 32–224 (true aspect kept,
   square 0.174 m cells) — source raster is the georeferenced 515×1150 grid.
2. Normalization clip is [40, 170] dB (not [40, 150]): this floor's physics
   tops out ~170 dB and the doc's window would clip ~50% of cells.
3. Furniture is per-meter, not 1 dB/crossing (per-crossing explodes at
   0.17 m cells: ~50 runs on an open-plan ray).
4. Resampling is max-loss pooling, not nearest-neighbor (labels stay
   integers; 2-px walls survive).

## Error bounds

- vs simulator ground truth: fill in after Phase C (target ≤ 3 dB RMSE,
  mean ± std over seeds {0,1,2}).
- vs reality: unknown until Phase D (target ≤ 8 dB RMSE on held-out
  measurement points — the physical floor set by shadow fading).

## Version / integrity

manifest `sim-v1.0`; grid and walkable-mask SHA-256 prefixes are recorded in
manifest.json and must match between dataset, training run, and web deploy.
