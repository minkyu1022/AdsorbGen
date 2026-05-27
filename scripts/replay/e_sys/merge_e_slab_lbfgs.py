#!/usr/bin/env python
"""Merge L-BFGS pristine slab references and materialize sid-keyed cache."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=24)
    ap.add_argument("--sid-index", default="/home/irteam/results/pristine_slabs/is2res.sid_index.pkl")
    ap.add_argument("--out-dir", default="/home/irteam/data/replay")
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--relaxed-pristine-out", default="")
    ap.add_argument("--require-all-shards", action="store_true")
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_slab = {}
    missing = []
    for shard in range(args.num_shards):
        path = shard_dir / f"e_slab_only_lbfgs_shard{shard}.pkl"
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open("rb") as f:
            part = pickle.load(f)
        by_slab.update(part)
        print(f"loaded slab shard {shard}: {len(part)} records")
    if missing and args.require_all_shards:
        raise FileNotFoundError("missing shard files:\n" + "\n".join(missing))
    if not by_slab:
        raise RuntimeError("no slab L-BFGS records loaded")

    with open(args.sid_index, "rb") as f:
        sid_to_slab = pickle.load(f)

    by_sid = {}
    missing_sids = 0
    for sid, slab_key in sid_to_slab.items():
        rec = by_slab.get(slab_key)
        if rec is None:
            missing_sids += 1
            continue
        by_sid[int(sid)] = {
            "e_total": float(rec["e_total"]),
            "slab_key": slab_key,
            "converged": bool(rec.get("converged", False)),
            "fmax": float(rec.get("fmax", float("nan"))),
            "n_steps": int(rec.get("n_steps", 0) or 0),
            "n_atoms": int(rec.get("n_atoms", 0) or 0),
            "source": f"E_slab_only_lbfgs_by_slab.pkl/{args.uma_model}/{args.uma_task}/ase.LBFGS",
        }

    slab_path = out_dir / "E_slab_only_lbfgs_by_slab.pkl"
    sid_path = out_dir / "E_slab_only_lbfgs.pkl"
    summary_path = out_dir / "E_slab_only_lbfgs_summary.json"
    with slab_path.open("wb") as f:
        pickle.dump(by_slab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with sid_path.open("wb") as f:
        pickle.dump(by_sid, f, protocol=pickle.HIGHEST_PROTOCOL)
    relaxed_pristine_path = None
    if args.relaxed_pristine_out:
        relaxed_pristine_path = Path(args.relaxed_pristine_out)
        relaxed_pristine_path.parent.mkdir(parents=True, exist_ok=True)
        relaxed_db = {}
        for key, rec in by_slab.items():
            out_rec = dict(rec)
            if rec.get("pos_relaxed") is not None:
                out_rec["pos"] = rec["pos_relaxed"]
            relaxed_db[key] = out_rec
        with relaxed_pristine_path.open("wb") as f:
            pickle.dump(relaxed_db, f, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "source": f"{args.uma_model}/{args.uma_task}/ase.LBFGS",
        "uma_model": args.uma_model,
        "uma_task": args.uma_task,
        "relaxer": "ase.LBFGS",
        "slab_records": len(by_slab),
        "sid_records": len(by_sid),
        "converged_slabs": sum(1 for r in by_slab.values() if r.get("converged")),
        "missing_sid_records": missing_sids,
        "missing_shards": missing,
        "E_slab_only_lbfgs": str(sid_path),
        "E_slab_only_lbfgs_by_slab": str(slab_path),
        "relaxed_pristine_slabs": str(relaxed_pristine_path) if relaxed_pristine_path else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
