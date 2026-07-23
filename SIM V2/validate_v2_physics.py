#!/usr/bin/env python3
"""
Cross-validation test case for the v2 enhanced-MK physics.

Runs the full engine (with the effective-obstruction calibration read from
manifest_v2) over several random transmitters and asserts the resulting
indoor path loss is physically plausible against the defensible references:

  - ITU-R P.1238 indoor-office plausibility (median in a physical band);
  - monotonic increase with frequency and a physical frequency slope;
  - preserved dynamic range (structure not flattened by the cap);
  - the scanner ABSOLUTE bound: an indoor 23 dBm Tx must give a median RSRP
    at least as good as the measured outdoor-donor median (-93 dBm @2600),
    since an indoor source is closer/stronger than a distant macro.

Note on the scanner: the per-band median RSRP is donor-dominated (adjacent
617/627 MHz channels differ by 23 dB = different cells, not propagation), so
it cannot pin a quantitative frequency slope. The defensible scanner metric is
the absolute plausibility bound above. Rigorous per-material calibration needs
a known-Tx walk (Phase D); these constants are ITU-anchored placeholders.

usage: python "SIM V2/validate_v2_physics.py" --repo . --device cpu [--n-tx 5]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# physical acceptance windows for indoor MEDIAN path loss (dB), per band.
# Lower bound ~ ITU-R P.1238 office; upper allows this dense floor to be
# moderately lossier. These are the "truth-worthy" gates.
TARGET_WINDOW = {
    619: (70, 100), 627: (70, 100), 1935: (90, 120), 2442: (95, 125),
    2510: (95, 128), 2600: (96, 130), 3500: (100, 140), 5500: (108, 150),
    6125: (110, 155),
}
SCANNER_OUTDOOR_MEDIAN_2600_DBM = -93.0   # measured, n41 outdoor donor
TX_POWER_DBM = 23.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-tx", type=int, default=5)
    ap.add_argument("--sim-v2", default="SIM V2")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    sd = repo / a.sim_v2
    sys.path.insert(0, str(repo / "SIM"))
    import engine_v2_torch as ET
    import torch

    grid = np.load(sd / "grid_model_v2.npy")
    inside = np.load(sd / "inside_mask_v2.npy")
    wk = np.load(sd / "walkable_mask_v2.npy")
    man = json.loads((sd / "manifest_v2.json").read_text())
    cell, freqs = man["cell_size_m"], man["freqs_mhz"]
    phys = man["physics"]

    sc = ET.TorchScene(grid, inside, cell, freqs_mhz=freqs, device=a.device,
                       n_relay_cache=12,
                       obs_solidity=phys.get("obs_solidity", 1.0),
                       obs_ceiling_db=phys.get("obs_ceiling_db", 0.0))
    ys, xs = np.nonzero(wk)
    rng = np.random.default_rng(1)
    maps = []
    for k in rng.choice(len(xs), a.n_tx, replace=False):
        with torch.no_grad():
            pl = sc.pathloss_maps((float(xs[k]), float(ys[k]))).cpu().numpy()
        maps.append(pl[:, inside])
    P = np.concatenate(maps, 1)                       # (nf, n_tx*n_indoor)

    med = np.array([np.median(P[i]) for i in range(len(freqs))])
    fails = []
    print(f"{'band':>6} {'p10':>5} {'p50':>5} {'p90':>5} {'window':>12} {'ok':>4}")
    for i, f in enumerate(freqs):
        lo, hi = TARGET_WINDOW[int(f)]
        ok = lo <= med[i] <= hi
        if not ok:
            fails.append(f"median @{f:.0f} = {med[i]:.0f} outside [{lo},{hi}]")
        print(f"{f:6.0f} {np.percentile(P[i],10):5.0f} {med[i]:5.0f} "
              f"{np.percentile(P[i],90):5.0f} {f'[{lo},{hi}]':>12} {'Y' if ok else 'N':>4}")

    mono = all(med[i] <= med[i + 1] + 1 for i in range(len(freqs) - 1))
    slope = med[-1] - med[0]
    dyn = np.percentile(P[6], 90) - np.percentile(P[6], 10)
    rsrp2600 = TX_POWER_DBM - med[5]
    clip = max(100 * (P[i] >= man["norm"]["pl_min_db"] +
                      man["norm"]["pl_range_db"] - 0.5).mean()
               for i in range(len(freqs)))

    print(f"\nmonotonic in frequency: {mono}")
    print(f"frequency slope 619->6125: {slope:.0f} dB (physical ~35-55)")
    print(f"dynamic range @3500 (p90-p10): {dyn:.0f} dB (>30 = structure kept)")
    print(f"max clip fraction: {clip:.2f}% (<1 required)")
    print(f"scanner bound: indoor 23 dBm Tx median RSRP @2600 = {rsrp2600:.0f} "
          f"dBm (must beat outdoor donor {SCANNER_OUTDOOR_MEDIAN_2600_DBM:.0f})")

    if not mono:
        fails.append("non-monotonic in frequency")
    if not (35 <= slope <= 60):
        fails.append(f"frequency slope {slope:.0f} unphysical")
    if dyn < 30:
        fails.append(f"dynamic range {dyn:.0f} too small (flattened)")
    if clip >= 1.0:
        fails.append(f"clip {clip:.1f}% too high")
    if rsrp2600 < SCANNER_OUTDOOR_MEDIAN_2600_DBM:
        fails.append("indoor Tx worse than outdoor donor (implausible)")

    if fails:
        print("\nVALIDATION FAILED:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("\nV2 PHYSICS VALIDATION PASSED — indoor path loss physically plausible")


if __name__ == "__main__":
    main()
