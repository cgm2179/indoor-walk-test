# SIM — UNet Path-Loss Surrogate Pipeline

Implements the "Design Context & Build Instructions" v1.0 document, phase by
phase. `manifest.json` is the single source of truth (R6): the Python
generator, the training notebook, and the browser Simulator tab all read
their constants from it.

## Phase map

| Phase | File | Status |
|---|---|---|
| A — physics generator | `phase_a.py` | ✅ tests pass, 0.5 s/map (target ≤2 s) |
| B — dataset (10,000 pairs) | `phase_b_dataset.py` | ✅ generator + splits + leakage audit |
| C — UNet training | `phase_c_train_colab.ipynb` | ▶ run on Colab (GPU) |
| D — calibration vs G-flex | `phase_d_calibrate.py` | ⛔ blocked: needs walk with known Tx |
| E — placement optimizer | `phase_e_optimizer.py` | ✅ + in-browser version |
| F — export & Simulator tab | `export_web_assets.py`, `web/` | ✅ live in Frontend_Data_Display.html |

**Next generation**: `MODEL_CARD_v2.md` is the planned sim-v2.0 (enhanced
Motley-Keenan: frequency exponent, angle of incidence, thickness, diffraction;
20 dB low-E facade; 9-band frequency union; co-channel interference mapping;
outdoor OSM sandbox foundation). v2 physics config is scaffolded in
`phase_a.py` under `V2` (inert until `PHYSICS_VARIANT="v2.0"`).

## Quick start

```bash
make everything        # prepare grid + tests + 10k dataset (~80 min) + assets
make test              # A.4 unit tests + D.3 sanity checks only
```

Open `Frontend_Data_Display.html` → **Simulator** tab. It works immediately
with the exact JS physics engine (~2–4 s per Solve). To get millisecond
solves, train the surrogate:

1. Upload `SIM/dataset/` to Google Drive as `indoor-walk-test-dataset/`
   (~0.5 GB; it is gitignored).
2. Run `SIM/phase_c_train_colab.ipynb` on Colab (GPU). It trains with
   seeds {0,1,2}, compares against FSPL / fitted log-distance / 3GPP 38.901
   InH baselines, and exports `pl_unet.onnx` after a ≤0.1 dB parity check.
3. Drop the file at `SIM/web/pl_unet.onnx`. The tab auto-upgrades
   (engine badge switches from "physics" to "UNet surrogate").

**Getting the trained model into any clone** (it is 124 MB — over GitHub's
100 MB in-repo limit, so it ships as a release asset instead; fp16 shrinking
was tried and FAILED the 0.1 dB parity gate at 0.38 dB, so full precision it
is): run `make model`, or
`gh release download surrogate-v1 -p pl_unet.onnx -O SIM/web/pl_unet.onnx`.
Backups also exist on Drive (`MyDrive/SIM/checkpoints/pl_unet.onnx`).

## Simulator tab (Phase F)

- Three sub-tabs: **Received Signal** (outdoor BS plane-wave, precomputed at
  8 bearings — the bearing field snaps to 45°), **Transmitter** (click to
  place, R8 snaps to walkable cells), **Combined** (linear power sum, R7).
- R10: Solve stays disabled until band / Tx power / gain (or bearing / P_ref)
  are set; no silent defaults.
- Every map has Static ⇄ **Time-lapse** (first-arrival animation, T = d/c,
  plane-wave staggering for the BS, ×10⁻⁸ time scale labeled on screen).
- PNG + CSV export (both grids, with metadata header).
- **Optimize placement**: exhaustive search over walkable cells (stride
  selectable), objectives = coverage with z·σ_SF fade margin / mean PL /
  hole-filling, returns clickable top-5. With the physics engine expect
  minutes (a confirm dialog shows the ETA); with the ONNX surrogate, seconds.

## Phase D unblocking (the riskiest item — schedule early)

Walk the floor with a transmitter of KNOWN power at a KNOWN position
(e.g. a green-pin location), logging ≥200 (x, y, RSRP, freq) points across
corridors, offices, and core shadows. Save as
`SIM/measurements_known_tx.csv` (columns: x_px, y_px, rsrp_dbm, freq_mhz —
model-grid cells), set `TX_KNOWN` in `phase_d_calibrate.py`, then
`python SIM/phase_d_calibrate.py --fit`. It least-squares fits
{n, per-material losses} bounded to ±50%, writes `calibration.json`, and the
README of that file says: regenerate Phase B and fine-tune 20 epochs.
