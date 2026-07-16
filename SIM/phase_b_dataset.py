#!/usr/bin/env python3
"""
Phase B — Dataset generation (spec §4).

- 2,500 Tx positions uniform over the walkable mask (R8), min spacing 2 cells
- 4 frequencies per position -> 10,000 (input, target) pairs
- 50% of samples get wall-loss jitter U(0.8, 1.2) per material (recorded)
- Targets: PL clipped to [pl_min, pl_min + pl_range] from the manifest, then
  normalized to [0, 1]; stored float16
- Splits BY POSITION (R4), stratified by floor octant, seed 0, committed to
  splits.json and never regenerated (the script refuses to overwrite it)
- Shards of 500 samples (125 positions x 4 freqs), resumable: existing shards
  are skipped, so the run can be interrupted and restarted freely

usage:
  python SIM/phase_b_dataset.py                  # full 10,000-sample run
  python SIM/phase_b_dataset.py --smoke          # 12 positions, quick check
  python SIM/phase_b_dataset.py --audit          # leakage + histogram checks
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pa", SIM / "phase_a.py")
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)

N_POSITIONS = 2500
MIN_SPACING = 2.0
SHARD_POS = 125                      # x4 freqs = 500 samples/shard
SEED = 0                             # B.4: fixed, never regenerate
JITTER_FRAC = 0.5


def sample_positions(walkable, rng):
    """Uniform over walkable cells with minimum spacing (grid-hash reject)."""
    cells = np.argwhere(walkable)    # (y, x)
    rng.shuffle(cells)
    taken = np.zeros(walkable.shape, bool)
    out = []
    r = int(np.ceil(MIN_SPACING))
    for y, x in cells:
        y0, y1 = max(0, y - r), min(walkable.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(walkable.shape[1], x + r + 1)
        if taken[y0:y1, x0:x1].any():
            continue
        taken[y, x] = True
        out.append((int(x), int(y)))
        if len(out) == N_POSITIONS:
            break
    return np.array(out)             # (N, 2) as (x, y)


def octant_of(pos, inside):
    ys, xs = np.nonzero(inside)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    ox = np.clip(((pos[:, 0] - x0) * 4) // max(x1 - x0, 1), 0, 3)
    oy = np.clip(((pos[:, 1] - y0) * 2) // max(y1 - y0, 1), 0, 1)
    return (oy * 4 + ox).astype(int)


def make_splits(pos, inside, rng):
    """2000/250/250 positions, stratified by octant (B.4)."""
    octs = octant_of(pos, inside)
    idx = {"train": [], "val": [], "test": []}
    for o in range(8):
        ids = np.nonzero(octs == o)[0]
        rng.shuffle(ids)
        n = len(ids)
        n_val, n_test = round(n * 0.1), round(n * 0.1)
        idx["val"] += ids[:n_val].tolist()
        idx["test"] += ids[n_val:n_val + n_test].tolist()
        idx["train"] += ids[n_val + n_test:].tolist()
    return {k: sorted(v) for k, v in idx.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--shard-mod", type=int, default=1,
                    help="parallel workers: process shards where s %% mod == rem")
    ap.add_argument("--shard-rem", type=int, default=0)
    args = ap.parse_args()

    manifest = json.loads((SIM / "manifest.json").read_text())
    grid = np.load(SIM / "grid_model.npy")
    inside = np.load(SIM / "inside_mask.npy")
    walkable = np.load(SIM / "walkable_mask.npy")
    cell = manifest["cell_size_m"]
    norm = manifest["norm"]
    pl_lo, pl_rng = norm["pl_min_db"], norm["pl_range_db"]
    f_lo, f_hi = np.log10(norm["freq_log_lo_mhz"]), np.log10(norm["freq_log_hi_mhz"])
    freqs = manifest["freqs_mhz"]
    out = SIM / "dataset"
    out.mkdir(exist_ok=True)

    if args.audit:
        return audit(out, inside)

    rng = np.random.default_rng(SEED)
    pos = sample_positions(walkable, rng)
    n_pos = 12 if args.smoke else N_POSITIONS
    pos = pos[:n_pos]

    split_file = out / "splits.json"
    if split_file.exists() and not args.smoke:
        splits = json.loads(split_file.read_text())   # B.4: never regenerate
    else:
        splits = make_splits(pos, inside, np.random.default_rng(SEED + 1))
        if not args.smoke:
            split_file.write_text(json.dumps(
                dict(seed=SEED, n_positions=n_pos, positions=pos.tolist(),
                     **splits), indent=0))

    jit_rng = np.random.default_rng(SEED + 2)
    # pre-draw jitter decisions for reproducibility independent of resume point
    jitter_flag = jit_rng.random((n_pos, len(freqs))) < JITTER_FRAC
    jitter_vals = jit_rng.uniform(0.8, 1.2, size=(n_pos, len(freqs), 6)).astype(np.float32)
    jitter_vals[~jitter_flag] = 1.0
    jitter_vals[:, :, 0] = 1.0                        # air never jitters

    n_shards = int(np.ceil(n_pos / SHARD_POS))
    for s in range(n_shards):
        if s % args.shard_mod != args.shard_rem:
            continue
        shard_path = out / f"shard_{s:03d}.npz"
        if shard_path.exists():
            print(f"shard {s}: exists, skipping")
            continue
        p0, p1 = s * SHARD_POS, min((s + 1) * SHARD_POS, n_pos)
        tx_l, ff_l, tg_l, jv_l, id_l = [], [], [], [], []
        for pi in range(p0, p1):
            x, y = pos[pi]
            for fi, f in enumerate(freqs):
                jv = jitter_vals[pi, fi]
                pl = pa.pathloss_map(grid, (float(x), float(y)), float(f),
                                     cell, jitter=jv)
                tgt = (np.clip(pl, pl_lo, pl_lo + pl_rng) - pl_lo) / pl_rng
                tx_l.append((x, y))
                ff_l.append((np.log10(f) - f_lo) / (f_hi - f_lo))
                tg_l.append(tgt.astype(np.float16))
                jv_l.append(jv)
                id_l.append(pi)
        np.savez_compressed(
            shard_path,
            tx_pos=np.array(tx_l, np.int16), freq_feat=np.array(ff_l, np.float32),
            target=np.stack(tg_l), jitter=np.stack(jv_l),
            pos_id=np.array(id_l, np.int32))
        print(f"shard {s + 1}/{n_shards} written ({p1 - p0} positions)")

    (out / "dataset_meta.json").write_text(json.dumps(dict(
        n_positions=n_pos, freqs_mhz=freqs, seed=SEED,
        jitter_frac=JITTER_FRAC, shards=n_shards,
        manifest_grid_sha=manifest["grid_sha256"],
        norm=norm), indent=2))
    print("dataset complete")


def audit(out, inside):
    """B.6: leakage audit + target histogram."""
    splits = json.loads((out / "splits.json").read_text())
    tr, va, te = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert not (tr & va or tr & te or va & te), "LEAKAGE: overlapping positions"
    print(f"splits disjoint: {len(tr)}/{len(va)}/{len(te)} positions")

    seen = {}
    hist = np.zeros(20)
    lo_clip = hi_clip = tot = 0
    for shard in sorted(out.glob("shard_*.npz")):
        d = np.load(shard)
        for pid in d["pos_id"]:
            seen.setdefault(int(pid), 0)
            seen[int(pid)] += 1
        t = d["target"].astype(np.float32)
        hist += np.histogram(t, bins=20, range=(0, 1))[0]
        lo_clip += (t <= 0).sum(); hi_clip += (t >= 1).sum(); tot += t.size
    for pid, cnt in seen.items():
        in_splits = (pid in tr) + (pid in va) + (pid in te)
        assert in_splits == 1, f"position {pid} in {in_splits} splits"
        assert cnt % 4 == 0, f"position {pid} has {cnt} samples (freqs split?)"
    print(f"all {len(seen)} positions in exactly one split, all freqs together")
    print("target histogram (20 bins):", (hist / hist.sum() * 100).round(1).tolist())
    print(f"at clip bounds: lo {100 * lo_clip / tot:.2f}%  hi {100 * hi_clip / tot:.2f}%")


if __name__ == "__main__":
    main()
