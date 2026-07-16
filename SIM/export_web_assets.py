#!/usr/bin/env python3
"""
Export browser assets for the Simulator tab in Frontend_Data_Display.html.

Writes SIM/web/sim_assets.js — a plain <script src>-loadable file (works on
file:// where fetch() does not) defining window.SIM_ASSETS with:
  manifest       the R6 manifest, inlined verbatim
  grid_b64       uint8 material grid (H*W)
  walkable_b64   uint8 walkable mask
  inside_b64     uint8 inside mask
  bs             precomputed outdoor-BS maps at 3.5 GHz for 8 bearings
                 (gain re P_ref as int16 deci-dB; arrival as uint16 ns)
                 -- the UI snaps the bearing field to the nearest 45 degrees

usage: python SIM/export_web_assets.py [--skip-bs]
"""
import argparse
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pa", SIM / "phase_a.py")
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)

BEARINGS = list(range(0, 360, 45))
BS_FREQ = 3500.0


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bs", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((SIM / "manifest.json").read_text())
    grid = np.load(SIM / "grid_model.npy")
    inside = np.load(SIM / "inside_mask.npy")
    walkable = np.load(SIM / "walkable_mask.npy")
    cell = manifest["cell_size_m"]

    bs = None
    if not args.skip_bs:
        gains, times = [], []
        for b in BEARINGS:
            print(f"BS bearing {b} deg ...")
            g, t = pa.bs_maps(grid, inside, float(b), BS_FREQ, cell)
            gains.append(np.clip(g * 10, -32000, 32000).astype(np.int16))
            t_ns = np.where(np.isfinite(t), t * 1e9, 65535)
            times.append(np.clip(t_ns, 0, 65535).astype(np.uint16))
        bs = dict(bearings=BEARINGS, f_mhz=BS_FREQ,
                  o2i_db=pa.O2I_DB["low_loss"],
                  gain_decidb_b64=[b64(g) for g in gains],
                  time_ns_b64=[b64(t) for t in times])

    out = dict(manifest=manifest,
               grid_b64=b64(grid.astype(np.uint8)),
               walkable_b64=b64(walkable.astype(np.uint8)),
               inside_b64=b64(inside.astype(np.uint8)),
               bs=bs)
    (SIM / "web").mkdir(exist_ok=True)
    path = SIM / "web" / "sim_assets.js"
    path.write_text("window.SIM_ASSETS = " + json.dumps(out) + ";\n")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
