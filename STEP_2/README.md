# STEP 2 — Log-distance + Multi-wall (Motley-Keenan) Baseline

Predicts received power at every cell of the STEP_1 grid:

```
PL(cell) = FSPL(1 m, f) + 10·n·log10(d) + saturate( Σ wall crossings + Σ clutter·meters )
Prx      = EIRP − PL
```

## Run it

```bash
python STEP_2/motley_keenan.py                    # defaults: 3500 MHz, n=2, EIRP 23 dBm
python STEP_2/motley_keenan.py --freq-mhz 1900 --eirp-dbm 30
```

~14 s for both transmitters (pure numpy, no GPU). Outputs land here:
`pathloss_tx<i>.npy` (dB), `heatmap_tx<i>.png`, `heatmap_best_server.png`,
`sim_params.json`, plus the consolidated grid and outside mask.

## Model decisions (each one was validated against a failure)

- **n = 2.0 (free space), not ~3**: walls are charged explicitly, so an
  office-grade exponent would double-count the environment. This is standard
  Motley-Keenan practice.
- **Wall crossings count contiguous runs once** — wall thickness in pixels
  doesn't inflate loss. Rays are sampled at 0.6 px so 2-px walls can't be
  stepped over.
- **Furniture is bulk clutter (0.3 dB/m of path)**, not per-crossing: a ray
  through open-plan office crosses dozens of drawn objects; per-crossing
  counting added ~50 dB of nonsense.
- **Core service areas are solid regions** (STEP_1 fills the hatching),
  otherwise each hatch stroke counted as a separate 20 dB wall.
- **Total obstruction loss saturates at `--wall-sat-db` (60 dB)**: straight
  rays over-punish deep shadow — real energy reroutes through corridors
  (COST-231 found multi-wall loss grows sublinearly). Set it huge to disable.
- **Cells outside the envelope are masked** from stats and grayed in renders.

## Sanity numbers (defaults, single Tx)

Indoor median ≈ −93 dBm, 10th percentile ≈ −108 dBm, ~64% of the floor above
−100 dBm — realistic for one 23 dBm small cell at 3.5 GHz on a 78 × 35 m
floor plate.

## Known limitations

2D straight rays: no reflections, no diffraction, no corridor waveguiding
(only proxied by the saturation), no floor/ceiling paths, doors drawn as
openings are sealed by wall consolidation. The visible ray streaks in the
heatmaps are the signature of this. STEP_3 (ray tracing / FDTD) is the
physics upgrade; STEP_4 calibrates the loss table against measurements.
