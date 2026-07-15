# STEP 4 — ML Layer

## 4b. Surrogate model (RadioUNet-style) — data generation WORKING

`generate_dataset.py` uses the STEP_2 engine as the ground-truth simulator to
produce (floor + Tx location → path-loss map) training pairs, with the Tx
sampled uniformly over indoor open cells:

```bash
python STEP_4/generate_dataset.py --n-samples 500   # ~8 min CPU, ~0.3 GB
```

Each `sample_NNNN.npz`: `x` = 3×257×575 float32 (wall dB / clutter dB-per-m /
log10 distance), `y` = 257×575 float16 path loss, plus the Tx position and
indoor mask. The `dataset/` folder is gitignored — regenerate at will (fixed
seed, deterministic).

Training (needs PyTorch, ideally GPU — Colab works): standard U-Net,
input `x`, regress `y`, L1 loss masked to indoor cells, ~80/20 split. The
payoff: millisecond heatmaps for any Tx position, enabling interactive
placement optimization (drag-the-transmitter in the browser, Step-2 quality).

## 4c. Calibration against walk-test data — BLOCKED, here's why and the fix

The walk-test scanner recorded **outdoor macro donors** (T-Mobile, PCI/band
in the data) — not a transmitter at a known indoor location. Calibrating the
material losses requires computing predicted loss Tx→measurement-point, and
the Tx (the donor site) location is unknown; GPS also drifted for most points
(only 2,409 of 10,248 land on the floor plate — see `walk.on_floor` in
`MATLAB/indoor_walk_test.mat`).

Two unblocking options:

1. **Best**: re-walk with a known transmitter — a hotspot/AP/CBRS radio at
   one of the green-pin locations — logging RSRP along a *known route*
   (waypoint-click instead of GPS). Then fit
   `argmin_θ Σ (RSRP_meas − (EIRP − PL_θ))²` over
   θ = {per-material losses, wall_sat, EIRP offset} — a 30-line scipy
   `least_squares` on top of STEP_2. The fit machinery can be written the
   day the data exists.
2. Obtain donor cell-site coordinates (e.g. from tower databases / FCC ASR
   for the PCIs in the data) and calibrate on the outdoor-to-indoor link —
   noisier, but no new walk needed.

## 4a. U-Net floor-plan classifier (CubiCasa5k) — deferred

Generalizing STEP_1 to arbitrary floor plans is a training project (GPU,
dataset download, ~a day of iteration). The manual STEP_1 pipeline is more
accurate for this one floor — exactly as the project plan predicted.
