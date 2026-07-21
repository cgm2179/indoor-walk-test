# SIM V2 — next-generation work (sim-v2.0, "Indoor Walk Test v3")

Home for the **planned** next generation: enhanced Motley-Keenan physics,
20 dB low-E facade, 9-band frequency union, co-channel interference mapping,
and the outdoor OSM sandbox foundation around FCC HQ (45 L Street NE).

Nothing here is trained yet. The deployed, validated model is **v1**
(`sim-v1.1`, test RMSE 4.68 dB), and it still lives in and runs from `SIM/`.

## Contents

| File | What it is |
|---|---|
| `MODEL_CARD_V2.md` | The v2.0 model card + full workflow (read this first). |
| `phase_c_train_colab_v3.ipynb` | The current training notebook (produced `surrogate-v1`). Reused for v2 with `IN_CH` bumped for the 7-class glass split. |
| `phase_c_train_colab_v2.ipynb`, `phase_c_train_colab.ipynb` | Superseded earlier training notebooks, kept for history. |
| `rerasterized_map_v1_baseline.png` | The v1 rasterization preview, as the visual starting point for v2's re-rasterization (workflow step 1). |

## Important: what did NOT move (and why)

The **live pipeline stays in `SIM/`**. `grid_model.npy`, `manifest.json`,
`walkable_mask.npy`, the phase A/B/D/E scripts, and `web/` are read by five
scripts and the deployed dashboard — moving them would break v1. v2 will
generate its **own** grid/manifest/dataset when the enhanced-MK physics and
the glass-class un-fold are implemented (`SIM/phase_a.py` holds the inert `V2`
config behind `PHYSICS_VARIANT`). Only then do those v2 artifacts belong here.

## Colab

The training notebook moved, so its Colab URL is now (note `%20` for the
space in the folder name):

```
https://colab.research.google.com/github/cgm2179/indoor-walk-test/blob/main/SIM%20V2/phase_c_train_colab_v3.ipynb
```

The notebook clones the repo and reads repo-root-relative paths
(`SIM/manifest.json`, `SIM/dataset`, …), so its new location does not change
what it reads — until v2 repoints those to this folder's own grid.

*Tip: the space in "SIM V2" becomes `%20` in URLs and needs quoting in the
shell. If that gets annoying, ask and I'll rename it to `SIM_V2`.*
