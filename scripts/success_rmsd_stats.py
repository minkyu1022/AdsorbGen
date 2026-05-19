#!/usr/bin/env python
"""Pairwise-RMSD distribution stats for multi-success systems, per stage.

For each system (sid) with >=2 success samples in a viz epoch dir, computes the
distribution of pairwise RMSD between samples at three pipeline stages:
  x0          - prior placement
  x1_flow     - flow model prediction
  x1_relaxed  - UMA relaxation end
both over all atoms and over adsorbate atoms only (tags from the buffer).
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
from ase.io import read


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def stats(vals):
    if not vals:
        return None
    v = np.asarray(vals, dtype=float)
    return dict(n=len(v), mean=float(v.mean()), std=float(v.std()),
                min=float(v.min()), median=float(np.median(v)), max=float(v.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--viz-ep-dir", required=True)
    p.add_argument("--stream-dir", required=True,
                   help="ReplayStream root (for adsorbate tags via buffer chunks)")
    args = p.parse_args()

    ep = Path(args.viz_ep_dir)

    # sid -> tags, from buffer chunk pkls
    sid_tags: dict = {}
    for f in glob.glob(f"{args.stream_dir}/shard_*/chunk_*.pkl"):
        with open(f, "rb") as fh:
            for e in pickle.load(fh):
                sid_tags.setdefault(int(e["sid"]), np.asarray(e["tags"]))

    idx = json.loads((ep / "_index.json").read_text())
    bysid: dict = {}
    for s in idx["systems"]:
        if s.get("success"):
            bysid.setdefault(int(s["sid"]), []).append(ep / s["sys_dir_name"])

    print(f"viz epoch: {ep}    success systems: {len(bysid)}\n")
    for sid in sorted(bysid):
        folders = bysid[sid]
        tags = sid_tags.get(sid)
        ads = np.where(tags == 2)[0] if tags is not None else None
        n_ads = len(ads) if ads is not None else "?"
        print(f"=== sid {sid}   success samples: {len(folders)}   adsorbate atoms: {n_ads} ===")
        if len(folders) < 2:
            print("  <2 success samples — pairwise RMSD distribution not defined\n")
            continue
        for kind in ("x0", "x1_flow", "x1_relaxed"):
            P = [read(str(fo / f"{kind}.pdb")).get_positions()
                 for fo in folders if (fo / f"{kind}.pdb").exists()]
            if len(P) < 2:
                print(f"  {kind:11s}: <2 files present")
                continue
            pairs = list(combinations(range(len(P)), 2))
            so = stats([rmsd(P[i], P[j]) for i, j in pairs])
            print(f"  {kind:11s} all-atom : mean={so['mean']:.3f} std={so['std']:.3f} "
                  f"min={so['min']:.3f} med={so['median']:.3f} max={so['max']:.3f} "
                  f"Å  (n_pairs={so['n']})")
            if ads is not None and len(ads) > 0:
                sa = stats([rmsd(P[i][ads], P[j][ads]) for i, j in pairs])
                print(f"  {'':11s} adsorbate: mean={sa['mean']:.3f} std={sa['std']:.3f} "
                      f"min={sa['min']:.3f} med={sa['median']:.3f} max={sa['max']:.3f} Å")
        print()


if __name__ == "__main__":
    main()
