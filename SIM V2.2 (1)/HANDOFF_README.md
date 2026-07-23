# v2 Enhanced Motley-Keenan — GPU dataset generation handoff

This bundle adds the **enhanced Motley-Keenan physics (sim-v2.0)** to your
indoor-walk-test project and a **GPU path** to generate the v2 dataset on Colab.
It does **not** train anything — training stays in `phase_c_v2_train.ipynb` on
your machine.

## What's here

```
SIM/
  physics_v2.py         Blocks A-D: P.2040 permittivity, Fresnel/slab (angle+
                        thickness), low-E resistive sheet, UTD diffraction, FAF,
                        7-class material table + fast per-crossing LUT.
                        ~30 physics self-tests:  python SIM/physics_v2.py --test
  engine_v2.py          Raster geometry (NumPy REFERENCE): crossing extraction
                        with incidence angle + thickness, all-band tracing, edge
                        detection, UTD shadow-fill (no saturation stand-in).
                        7 geometry tests:        python SIM/engine_v2.py --test
  engine_v2_torch.py    CUDA port of engine_v2, verified to match it.
                        Parity:  python SIM/engine_v2_torch.py --parity --device cuda
                        Bench:   python SIM/engine_v2_torch.py --bench  --device cuda
SIM V2/
  phase_a_v2.py         7-class grid un-fold (interior vs exterior glass) +
                        manifest_v2.json writer.  python "SIM V2/phase_a_v2.py" --test
  phase_b_v2_generate.py  Dataset generator (GPU/CPU), writes shards in the
                        SAME format as v1 phase_b so phase_c_v2 consumes them.
  phase_b_v2_generate_colab.ipynb   << RUN THIS ON COLAB (GPU) to generate >>
  phase_a_v2_enhanced_mk.ipynb   working physics demo (was a TODO scaffold):
                        material table, low-E facade, sub-secant angle law,
                        diffraction fill — all runnable.
  phase_b_v2_dataset.ipynb   thin local/CPU driver for the generator.
  phase_c_v2_train.ipynb   COMPLETE runnable trainer (was a stub): reads
                        manifest_v2, 10-channel input, v2-engine baselines,
                        ONNX export with parity gate.
docs/
  v2_physics.tex / .pdf  The full equation set (this is the .TeX you asked for).
```

## How to generate the dataset (the actual handoff)

1. Push these files to your `indoor-walk-test` repo (keep the paths above).
2. Open **`SIM V2/phase_b_v2_generate_colab.ipynb`** in Colab, set Runtime -> GPU.
3. Run the cells top to bottom. They:
   - build the 7-class grid from your v1 grid + `material_overrides.json`,
   - **assert GPU/CPU parity** (stops if the engine is wrong),
   - smoke-test 12 positions, then generate all 2,500 x 9 bands to Drive,
   - audit (split disjointness, per-band clip fractions).
4. Then open `phase_c_v2_train.ipynb` and train on the Drive folder. It is a
   complete fork of your v3 trainer with a grid-sha tripwire, the 10-channel
   input, v2-engine baselines, and the ONNX parity gate.

   NOTE: the old phase_c_v2 scaffold read the *v1* manifest (`SIM/manifest.json`,
   clip [40,170]) while the v2 dataset is normalised with [40,230]. That
   mismatch would have silently corrupted training. The rewritten notebook
   reads `SIM V2/manifest_v2.json` and asserts the grid sha matches the dataset.

## Before you run: one grid prerequisite

The un-fold needs an **interior-glass (material id 6)** entry in
`STEP_1/material_overrides.json` for the lunch-room / meeting-room enclosure.
Without it, class 6 comes out empty (the grid is 7-class-capable but still
effectively 6-class). See MODEL_CARD_V2 sec 5.2.

## Key physics results worth knowing

- **Low-E facade**: R_s = 25 ohm/sq gives your measured 20 dB at normal
  incidence but **34 dB at 75 deg** — oblique macros penetrate far less than
  v1's fixed 15 dB O2I.  Strengthens the co-channel story.
- **Angle law**: the exact Fresnel slab is **sub-secant** (Snell refracts the
  ray toward the normal inside the wall).  A secant law over-attenuates oblique
  wall loss by ~40% at 56 deg, 170% at 80 deg.  The scaffold's todo-angle
  secant would have been wrong.
- **Saturation removed**: UTD supplies the deep-shadow energy v1 faked with a
  cap; cells above the clip ceiling drop from ~8% to ~3%.
- **Clip window widened** to [40, 230] dB: the frequency-scaled 5.5/6 GHz bands
  are genuinely much lossier (indoor p99.9 ~300 dB at 6 GHz).

## Verification status

All four suites pass: physics (30 tests), geometry (7), torch parity
(p99.99 < 0.05 dB vs NumPy over the training range), phase-A-v2 (3).
The GPU path is parity-clean on CPU; **run the parity cell on the actual GPU
first** — it's the gate before generating.

Every material constant is literature (P.2040 + fitted sub-resolution excess),
NOT yet calibrated to the building.  Phase D fits it once a known-Tx walk
exists; the `jitter` field in each shard makes that a fine-tune, not a retrain.
