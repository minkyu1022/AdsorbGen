#!/usr/bin/env python
"""Materialize UMA-s-1p2 bare-slab references from batched relabel shards."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--old-by-sid", default="/home1/irteam/data-vol1/minkyu/data/replay/E_slab_only_lbfgs.pkl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--require-all-shards", action="store_true")
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_slab = {}
    missing = []
    for shard in range(args.num_shards):
        path = shard_dir / f"bare_slab_pkl_shard{shard}.pkl"
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open("rb") as f:
            part = pickle.load(f)
        by_slab.update(part)
        print(f"loaded shard {shard}: {len(part)}")
    if missing and args.require_all_shards:
        raise FileNotFoundError("missing shard files:\n" + "\n".join(missing))
    if not by_slab:
        raise RuntimeError("no bare slab records loaded")

    with open(args.old_by_sid, "rb") as f:
        old_by_sid = pickle.load(f)

    by_sid = {}
    missing_sid = 0
    for sid, old in old_by_sid.items():
        slab_key = old.get("slab_key") if isinstance(old, dict) else None
        rec = by_slab.get(slab_key)
        if rec is None:
            missing_sid += 1
            continue
        by_sid[int(sid)] = {
            "e_total": float(rec["e_total"]),
            "E_bare_slab": float(rec["e_total"]),
            "slab_key": slab_key,
            "converged": bool(rec.get("converged", False)),
            "fmax": float(rec.get("fmax", float("nan"))),
            "n_steps": int(rec.get("n_steps", 0) or 0),
            "n_atoms": int(rec.get("n_atoms", 0) or 0),
            "source": "E_bare_slab_lbfgs_by_slab.pkl/uma-s-1p2/oc20/custom_batched_LBFGS",
        }

    relaxed_db = {}
    for key, rec in by_slab.items():
        out = dict(rec)
        out["pos"] = rec["pos_relaxed"]
        out["source_name"] = "bare_slab"
        relaxed_db[key] = out

    by_slab_path = out_dir / "E_bare_slab_lbfgs_by_slab.pkl"
    by_sid_path = out_dir / "E_bare_slab_lbfgs_by_sid.pkl"
    relaxed_path = out_dir / "bare_slabs_lbfgs.pkl"
    summary_path = out_dir / "E_bare_slab_lbfgs_summary.json"
    with by_slab_path.open("wb") as f:
        pickle.dump(by_slab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with by_sid_path.open("wb") as f:
        pickle.dump(by_sid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with relaxed_path.open("wb") as f:
        pickle.dump(relaxed_db, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "source": "uma-s-1p2/oc20/custom_batched_LBFGS/from_stored_pos_relaxed",
        "slab_records": len(by_slab),
        "sid_records": len(by_sid),
        "converged_slabs": sum(1 for r in by_slab.values() if r.get("converged")),
        "missing_sid_records": missing_sid,
        "missing_shards": missing,
        "E_bare_slab_lbfgs_by_slab": str(by_slab_path),
        "E_bare_slab_lbfgs_by_sid": str(by_sid_path),
        "bare_slabs_lbfgs": str(relaxed_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
