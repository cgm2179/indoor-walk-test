#!/usr/bin/env python3
"""
Package the floor data for the browser app (WEB/index.html).

Exports exactly the tensors the surrogate U-Net was trained on (same 2x
downsample, same 256x568 crop, same normalization constants), so the ONNX
model exported from the Colab notebook plugs in without any resampling:

  assets/wall_map.bin     float32 H*W  per-crossing wall loss (dB)
  assets/clutter_map.bin  float32 H*W  bulk clutter loss (dB/m)
  assets/indoor_mask.bin  uint8   H*W  1 = indoor
  assets/walls.png        transparent overlay of walls for display
  assets/meta.json        sizes, scales, normalization, band table, Tx pins

usage: python WEB/export_web_assets.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "gd", ROOT / "STEP_4" / "generate_dataset.py")
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)

CROP_H, CROP_W = 256, 568          # must match the training notebook
DS = 2


def main():
    out = ROOT / "WEB" / "assets"
    out.mkdir(parents=True, exist_ok=True)

    grid = np.load(ROOT / "STEP_2" / "material_grid_consolidated.npy")
    outside = np.load(ROOT / "STEP_2" / "outside_mask.npy")
    mats = json.loads((ROOT / "STEP_1" / "materials.json").read_text())
    meta1 = json.loads((ROOT / "STEP_1" / "floorplan_meta.json").read_text())
    txs = json.loads((ROOT / "STEP_1" / "transmitters.json").read_text())["transmitters"]

    loss_db = np.zeros(256, np.float32)
    loss_per_m = np.zeros(256, np.float32)
    for k, v in mats.items():
        loss_db[int(k)] = v["loss_db"]
        loss_per_m[int(k)] = v.get("loss_per_m_db", 0.0)

    g = gd.downsample_max_loss(grid, loss_db, DS)[:CROP_H, :CROP_W]
    ind = (~outside[: (grid.shape[0] // DS) * DS: DS,
                    : (grid.shape[1] // DS) * DS: DS])[:CROP_H, :CROP_W]

    loss_db[g].astype(np.float32).tofile(out / "wall_map.bin")
    loss_per_m[g].astype(np.float32).tofile(out / "clutter_map.bin")
    ind.astype(np.uint8).tofile(out / "indoor_mask.bin")

    walls = np.isin(g, [1, 2, 3, 5, 6, 7])
    rgba = np.zeros((CROP_H, CROP_W, 4), np.uint8)
    rgba[walls] = (20, 20, 30, 235)
    Image.fromarray(rgba).save(out / "walls.png")

    (out / "meta.json").write_text(json.dumps(dict(
        H=CROP_H, W=CROP_W,
        meters_per_px=meta1["meters_per_px"] * DS,
        x_scale=[20.0, 0.3, 3.0], y_scale=150.0,       # training normalization
        train_freq_mhz=3500.0,
        eirp_dbm=23.0, noise_figure_db=7.0,
        tx_default=[dict(x=t["x_px"] / DS, y=t["y_px"] / DS) for t in txs],
        bands=[
            dict(label="LTE B71 617 MHz", f_mhz=617.0, bw_mhz=10, scs_khz=15),
            dict(label="LTE B2 1960 MHz", f_mhz=1960.0, bw_mhz=20, scs_khz=15),
            dict(label="NR n41 2506 MHz", f_mhz=2506.0, bw_mhz=60, scs_khz=30),
            dict(label="NR n77 3750 MHz", f_mhz=3750.0, bw_mhz=100, scs_khz=30),
        ],
        note="band shift applied client-side: PL_f = PL_3500 + 20log10(f/3500) "
             "(exact for distance, approximate for wall slope)",
    ), indent=2))
    print(f"assets written to {out} ({CROP_W}x{CROP_H}, "
          f"{meta1['meters_per_px'] * DS:.4f} m/px)")


if __name__ == "__main__":
    main()
