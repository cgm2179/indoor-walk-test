# 7th-Floor Indoor RF Propagation — Digital Twin

What's **active** vs **archived**. If you're iterating, you're in the top half.

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
