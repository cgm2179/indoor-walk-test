# 7th-Floor Indoor RF Propagation — Digital Twin

What's **active** vs **archived**. If you're iterating, you're in the top half.
Before changing anything structural, read **[DECISIONS.md](DECISIONS.md)** —
every load-bearing choice with the reasoning and the failure that forced it,
plus the ranked list of known-wrong things. Rasterization improvement roadmap:
[STEP_1/README.md](STEP_1/README.md). Retraining with new physics or a new
grid: **[SIM/retrain_for_physics_training_map_v2.md](SIM/retrain_for_physics_training_map_v2.md)**
— the full runbook plus the complete v1 chronicle and error->fix table.
Next-generation plan (enhanced Motley-Keenan, 20 dB low-E facade, outdoor
foundation): **[SIM V2/MODEL_CARD_V2.md](SIM%20V2/MODEL_CARD_V2.md)**. The v2 phase notebooks (enhanced-MK physics, v2 dataset, v2 training)
live in `SIM V2/`; the v1 training notebooks stay in `SIM/`.

**Reproducing from a fresh clone**: `make model` (trained surrogate, release
`surrogate-v1`) · `make dataset-fetch` (training data, release `dataset-v1`)
or `make dataset` to regenerate · `make test` to verify physics · serve the
dashboard with VS Code Live Preview or `python3 -m http.server`.

## Active

| Path | What it is |
|---|---|
| **`SIM/`** | The current project: physics generator (Phase A), dataset (B), Colab training (`phase_c_train_colab_v3.ipynb`), calibration scaffold (D), optimizer (E), web assets (F). Start at [SIM/README.md](SIM/README.md). |
| **`Frontend_Data_Display.html`** | The dashboard — walk-test viewer + the **Simulator** tab. Serve with `python3 -m http.server 8432` and open in a browser. Needs `timeseries_data.js` and `SIM/web/` (both here). |
| `STEP_1/`, `STEP_2/` | Upstream inputs the SIM pipeline reads: the rasterized material grid, consolidation, walk-test heatmaps. Re-run only when the floor plan or material table changes. |
| `7th_Floor_...V2.2.png` + `.TAB` + `.aux.xml` + `_PseudoMercator.csv` | Source ground truth: the georeferenced floor plan and its QGIS control points. Everything derives from these. |
| `Makefile` | `make everything` / `make test` / `make dataset` / `make assets`. |

Current status: **surrogate deployed** — UNet v3 trained (test RMSE 4.68 dB /
MAE 2.87 dB vs simulator truth, beats FSPL / log-distance / 38.901 baselines;
single seed), exported to `SIM/web/pl_unet.onnx` (local only, gitignored;
backup on Drive under SIM/checkpoints/) and running in the Simulator tab ·
dataset (10k samples) on Drive · physics v1.2 (per-material frequency
scaling) queued behind a flag in `SIM/phase_a.py` · remaining: R9 seeds
{1,2} for mean±std, and Phase D calibration (blocked on a known-Tx walk).

## ARCHIVE/ (not iterated on — kept for provenance)

| Path | What it is |
|---|---|
| `raw_walk_data/` | The original scanner exports: `CSV/`, `DTR/`, `Device032409007/`, concatenated CSVs, misc. |
| `early_analysis/` | First-generation scripts/notebooks: CSV concatenation, histogram R script, methods notebook, dashboard builder. |
| `superseded/WEB/` | The standalone coverage app — replaced by the Simulator tab inside the dashboard. |
| `superseded/STEP_4/` | The first surrogate (8-cell notebook, small U-Net) — replaced by `SIM/` Phase C. |
| `superseded/STEP_3/` | Ray-tracing/FDTD upgrade notes — folded into the SIM physics ladder (`SIM/phase_a.py`, §12.3). |
| `MATLAB/` | The `.mat` data bundle + MATLAB ports. Still functional (paths patched); re-run `export_to_matlab.py` if STEP_1/2 outputs change. |

Nothing in ARCHIVE is imported by active code, with one exception: the MATLAB
exporter reads the concatenated walk CSV from `ARCHIVE/raw_walk_data/`.
