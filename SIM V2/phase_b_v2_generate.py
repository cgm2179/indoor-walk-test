#!/usr/bin/env python3
"""
Phase B v2 — enhanced-Motley-Keenan dataset generator (GPU or CPU).

Wraps the torch engine (SIM/engine_v2_torch.py) to produce the v2 training set:
2,500 walkable transmitter positions x 9 bands of enhanced-MK path loss, in the
SAME shard format as v1 phase_b so phase_c_v2_train consumes it unchanged.

The speedup vs v1: the engine traces geometry ONCE per transmitter and returns
all 9 bands together, on the GPU. Wall-loss jitter (the Phase-D fine-tune hook)
is applied per position via scene.set_material_scale.

Shards (shard_NNN.npz), v1-compatible:
  tx_pos   int16   (N,2)  cell (x,y)
  freq_feat float32 (N,)  (log10 f - log10 619)/(log10 6125 - log10 619)
  target   float16 (N,H,W) (clip(PL, pl_lo, pl_lo+pl_rng) - pl_lo)/pl_rng
  jitter   float32 (N,7)  per-material multiplier used (1.0 = none)
  pos_id   int32   (N,)   index into splits.json positions
splits.json: seed, positions, train/val/test (by position, octant-stratified).

usage (called by phase_b_v2_generate_colab.ipynb):
  python "SIM V2/phase_b_v2_generate.py" --smoke --device cuda \\
     --grid "SIM V2/grid_model_v2.npy" --inside "SIM V2/inside_mask_v2.npy" \\
     --walkable "SIM V2/walkable_mask_v2.npy" --manifest "SIM V2/manifest_v2.json" \\
     --out /content/smoke_v2
  python "SIM V2/phase_b_v2_generate.py" --audit --manifest ... --out ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

N_POSITIONS = 2500
MIN_SPACING = 2.0
SHARD_POS = 125          # x9 bands = 1125 samples/shard
SEED = 0
JITTER_FRAC = 0.5
N_CLASSES = 7


# ---- position sampling / splits (identical policy to v1 phase_b) -----------
def sample_positions(walkable, rng, n_target):
    cells = np.argwhere(walkable)          # (y,x)
    rng.shuffle(cells)
    taken = np.zeros(walkable.shape, bool)
    out, r = [], int(np.ceil(MIN_SPACING))
    for y, x in cells:
        y0, y1 = max(0, y - r), min(walkable.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(walkable.shape[1], x + r + 1)
        if taken[y0:y1, x0:x1].any():
            continue
        taken[y, x] = True
        out.append((int(x), int(y)))
        if len(out) == n_target:
            break
    return np.array(out)


def octant_of(pos, inside):
    ys, xs = np.nonzero(inside)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    ox = np.clip(((pos[:, 0] - x0) * 4) // max(x1 - x0, 1), 0, 3)
    oy = np.clip(((pos[:, 1] - y0) * 2) // max(y1 - y0, 1), 0, 1)
    return (oy * 4 + ox).astype(int)


def make_splits(pos, inside, rng):
    octs = octant_of(pos, inside)
    idx = {"train": [], "val": [], "test": []}
    for o in range(8):
        ids = np.nonzero(octs == o)[0]
        rng.shuffle(ids)
        nv = round(len(ids) * 0.1)
        idx["val"] += ids[:nv].tolist()
        idx["test"] += ids[nv:2 * nv].tolist()
        idx["train"] += ids[2 * nv:].tolist()
    return {k: sorted(v) for k, v in idx.items()}


def build_scene(grid, inside, cell, freqs, device, n_relay_cache=16,
                obs_solidity=1.0, obs_ceiling_db=0.0):
    import torch  # noqa
    import engine_v2_torch as ET
    return ET.TorchScene(grid, inside, cell, freqs_mhz=freqs, device=device,
                         n_relay_cache=n_relay_cache,
                         obs_solidity=obs_solidity, obs_ceiling_db=obs_ceiling_db)


def generate(args, manifest):
    import torch
    grid = np.load(args.grid)
    inside = np.load(args.inside)
    walkable = np.load(args.walkable)
    cell = manifest["cell_size_m"]
    freqs = manifest["freqs_mhz"]
    norm = manifest["norm"]
    pl_lo, pl_rng = norm["pl_min_db"], norm["pl_range_db"]
    f_lo = np.log10(norm["freq_log_lo_mhz"])
    f_hi = np.log10(norm["freq_log_hi_mhz"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    n_pos = 12 if args.smoke else N_POSITIONS
    rng = np.random.default_rng(SEED)
    pos = sample_positions(walkable, rng, n_pos)

    split_file = out / "splits.json"
    if split_file.exists() and not args.smoke:
        splits = json.loads(split_file.read_text())
    else:
        splits = make_splits(pos, inside, np.random.default_rng(SEED + 1))
        if not args.smoke:
            split_file.write_text(json.dumps(dict(
                seed=SEED, n_positions=n_pos, positions=pos.tolist(),
                grid_sha256=manifest.get("grid_sha256"), **splits), indent=0))

    jrng = np.random.default_rng(SEED + 2)
    jflag = jrng.random(n_pos) < JITTER_FRAC
    jvals = jrng.uniform(0.8, 1.2, size=(n_pos, N_CLASSES)).astype(np.float32)
    jvals[~jflag] = 1.0
    jvals[:, 0] = 1.0                                  # air never jitters

    phys = manifest.get("physics", {})
    scene = build_scene(grid, inside, cell, freqs, args.device,
                        n_relay_cache=(4 if args.smoke else 16),
                        obs_solidity=phys.get("obs_solidity", 1.0),
                        obs_ceiling_db=phys.get("obs_ceiling_db", 0.0))
    ff = ((np.log10(np.asarray(freqs)) - f_lo) / (f_hi - f_lo)).astype(np.float32)
    H, W = grid.shape

    n_shards = int(np.ceil(n_pos / SHARD_POS))
    for s in range(n_shards):
        if s % args.shard_mod != args.shard_rem:
            continue
        shard_path = out / f"shard_{s:03d}.npz"
        if shard_path.exists() and not args.smoke:
            print(f"shard {s}: exists, skip", flush=True)
            continue
        p0, p1 = s * SHARD_POS, min((s + 1) * SHARD_POS, n_pos)
        tx_l, ff_l, tg_l, jv_l, id_l = [], [], [], [], []
        for pi in range(p0, p1):
            x, y = pos[pi]
            jv = jvals[pi]
            if jflag[pi]:
                scene.set_material_scale(np.repeat(jv[:, None], len(freqs), 1))
            else:
                scene.set_material_scale(None)
            with torch.no_grad():
                pl = scene.pathloss_maps((float(x), float(y))).cpu().numpy()  # (9,H,W)
            tgt = ((np.clip(pl, pl_lo, pl_lo + pl_rng) - pl_lo) / pl_rng).astype(np.float16)
            for fi in range(len(freqs)):
                tx_l.append((x, y)); ff_l.append(ff[fi])
                tg_l.append(tgt[fi]); jv_l.append(jv); id_l.append(pi)
            print(f"  pos {pi + 1}/{n_pos}", flush=True)
        np.savez_compressed(
            shard_path, tx_pos=np.array(tx_l, np.int16),
            freq_feat=np.array(ff_l, np.float32), target=np.stack(tg_l),
            jitter=np.stack(jv_l), pos_id=np.array(id_l, np.int32))
        print(f"shard {s + 1}/{n_shards}: {p1 - p0} positions -> {shard_path.name}",
              flush=True)

    (out / "dataset_v2_meta.json").write_text(json.dumps(dict(
        n_positions=n_pos, freqs_mhz=freqs, seed=SEED, n_classes=N_CLASSES,
        clip_db=[pl_lo, pl_lo + pl_rng], grid_sha256=manifest.get("grid_sha256"),
        engine="enhanced_motley_keenan_v2"), indent=2))
    print("generation complete", flush=True)


def audit(args, manifest):
    out = Path(args.out)
    freqs = manifest["freqs_mhz"]
    pl_lo, pl_rng = manifest["norm"]["pl_min_db"], manifest["norm"]["pl_range_db"]
    splits = json.loads((out / "splits.json").read_text())
    tr, va, te = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert not (tr & va or tr & te or va & te), "LEAKAGE: overlapping positions"
    print(f"splits disjoint: {len(tr)}/{len(va)}/{len(te)} positions")
    if splits.get("grid_sha256") and manifest.get("grid_sha256"):
        assert splits["grid_sha256"] == manifest["grid_sha256"], \
            "grid sha mismatch: dataset was built from a different grid"
        print("grid sha matches manifest")

    seen, nband = {}, len(freqs)
    lo = np.zeros(nband); hi = np.zeros(nband); tot = np.zeros(nband)
    for shard in sorted(out.glob("shard_*.npz")):
        d = np.load(shard)
        for pid in d["pos_id"]:
            seen[int(pid)] = seen.get(int(pid), 0) + 1
        t = d["target"].astype(np.float32)
        # freq index recovered from the (sorted) freq_feat order within a position
        ffs = d["freq_feat"]
        uniq = np.unique(ffs)
        for k, f in enumerate(sorted(uniq)):
            m = np.isclose(ffs, f)
            tt = t[m]
            lo[k] += (tt <= 0).sum(); hi[k] += (tt >= 1).sum(); tot[k] += tt.size
    for pid, c in seen.items():
        assert c % nband == 0, f"position {pid} has {c} samples (freq split?)"
    print(f"all {len(seen)} positions have all {nband} bands together")
    print("per-band clip fractions (low% / high%):")
    for k, f in enumerate(freqs):
        print(f"  {f:7.0f} MHz: {100*lo[k]/max(tot[k],1):5.2f}% / "
              f"{100*hi[k]/max(tot[k],1):5.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid"); ap.add_argument("--inside")
    ap.add_argument("--walkable"); ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-mod", type=int, default=1)
    ap.add_argument("--shard-rem", type=int, default=0)
    args = ap.parse_args()
    # engine modules live in SIM/ (sibling of this SIM V2/ script)
    sys.path.insert(0, str(Path(args.manifest).resolve().parents[1] / "SIM"))
    manifest = json.loads(Path(args.manifest).read_text())
    if args.audit:
        return audit(args, manifest)
    generate(args, manifest)


if __name__ == "__main__":
    main()
