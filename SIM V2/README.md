# SIM V2 — next-generation work (sim-v2.0, "Indoor Walk Test v3")

Home for the **planned** next generation: enhanced Motley-Keenan physics,
20 dB low-E facade, 9-band frequency union, co-channel interference mapping,
and the outdoor OSM sandbox foundation around FCC HQ (45 L Street NE).

Nothing here is trained yet. The deployed, validated model is **v1**
(`sim-v1.1`, test RMSE 4.68 dB), which lives in and runs from `SIM/` — including
its training notebooks (`SIM/phase_c_train_colab*.ipynb`).

## Contents

| File | What it is |
|---|---|
| `MODEL_CARD_V2.md` | The v2.0 model card + full workflow. **Read this first.** |
| `phase_a_v2_enhanced_mk.ipynb` | **Phase A v2** — enhanced-MK physics dev: implements the 7-class glass un-fold and the γ frequency law, plots them, generates a sample map; scaffolds angle / thickness / diffraction / FAF as TODO cells with the math. This is where the v2 physics gets built. |
| `phase_b_v2_dataset.ipynb` | **Phase B v2** — v2 dataset (7-class, 9-band union). Scaffold: produces a real v2 set once `PHYSICS_VARIANT='v2.0'` is implemented. |
| `phase_c_v2_train.ipynb` | **Phase C v2** — training with `IN_CH=10` (7 classes + tx + freq + dist). A minimal fork of the v1 `SIM/phase_c_train_colab_v3.ipynb` — only the input builder and channel count differ. |
| `rerasterized_map_v1_baseline.png` | The v1 rasterization preview, as the visual starting point for v2's re-rasterization (workflow step 1). |

## Build order (see MODEL_CARD_V2 §11)

1. `phase_a_v2` → implement the enhanced-MK engine in `SIM/phase_a.py` behind
   `PHYSICS_VARIANT='v2.0'` (7-class grid, γ, then angle/thickness/diffraction);
   add unit tests.
2. `phase_b_v2` → generate the 9-band, 7-class dataset; audit.
3. `phase_c_v2` → train (`IN_CH=10`); baselines; ONNX export with parity gate.
4. Phase D (calibration) and deploy as `surrogate-v2` / `dataset-v2`.

## Important: what stays in `SIM/` (do not move)

The **live v1 pipeline** — `grid_model.npy`, `manifest.json`,
`walkable_mask.npy`, `phase_a/b/d/e` scripts, `web/`, and the v1 training
notebooks — is read by the deployed dashboard and five scripts. v2 generates
its **own** grid/manifest/dataset once the enhanced-MK physics and the glass
un-fold are implemented (`SIM/phase_a.py` holds the inert `V2` config behind
`PHYSICS_VARIANT`). Only those v2 outputs belong here.

## Colab URLs

```
https://colab.research.google.com/github/cgm2179/indoor-walk-test/blob/main/SIM%20V2/phase_a_v2_enhanced_mk.ipynb
https://colab.research.google.com/github/cgm2179/indoor-walk-test/blob/main/SIM%20V2/phase_b_v2_dataset.ipynb
https://colab.research.google.com/github/cgm2179/indoor-walk-test/blob/main/SIM%20V2/phase_c_v2_train.ipynb
```

*The space in "SIM V2" becomes `%20` in URLs and needs quoting in the shell.
If that gets annoying, ask and I'll rename it to `SIM_V2`.*
