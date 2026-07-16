# STEP 3 — Real Physics Upgrade (assessment: not a <20-minute step)

Per the project rule ("if Step 3 is quick, <20 min, do it; otherwise move
on"), Step 3 was assessed and **deferred** — every route needs something this
machine doesn't have today. Step 2's engine is the working baseline; Step 4
proceeds. Here is exactly what each route takes when you're ready.

## Route A — MATLAB ray tracing

Needs: MATLAB + Antenna Toolbox (R2020b+ for `raytrace` indoor).

1. Everything is already normalized in `MATLAB/indoor_walk_test.mat`
   (`load_indoor_walk_test.m` shows the frames; `motley_keenan.m` is the
   native port of the Step 2 model for cross-checking).
2. Indoor ray tracing wants 3D geometry (STL): extrude the wall cells of
   `floorplan.material_grid` to ~3 m height and feed `siteviewer(SceneModel=stl)`,
   then `propagationModel("raytracing", Method="sbr", MaxNumReflections=4)`
   with `txsite`/`rxsite` in cartesian coords (use the local-meter frame).
   Ask Claude for `grid_to_stl.py` when you get there — it's ~an hour of work.

## Route B — Sionna RT (Python, built for 5G NR research)

Needs: NVIDIA GPU (or patience on CPU), `pip install sionna` (pulls
TensorFlow — a heavyweight install; use a venv).

Scene geometry is Mitsuba XML: same extrusion need as Route A, with per-
material `itu_concrete`, `itu_glass`, `itu_metal` radio materials mapped from
the ids. Sionna then gives coverage maps with true reflection/diffraction and
channel impulse responses (delay spread — the metric from the project notes).

## Route C — 2D FDTD (the time-elapsed wave animation)

The sizing math at 3.5 GHz: λ = 85.7 mm, stable grid ≤ λ/10 ≈ **8.6 mm** →
78.1 m × 35.0 m = **9,100 × 4,080 ≈ 37 M cells**, ~15k timesteps (one floor
traversal at Courant limit) ≈ **10¹² cell-updates** — a GPU job (Taichi or
CuPy), minutes not seconds, and the material grid must be resampled 8×
denser. Downscaled demos (e.g. 700 MHz on a wing of the floor) are feasible
on CPU if you want the animation before getting GPU access.

## Why this order is fine

Motley-Keenan (Step 2) + measurement calibration (Step 4) is the standard
industrial planning stack; ray tracing/FDTD refine the same material grid and
Tx positions, so nothing done so far is throwaway.
