#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("ADSGEN_ROOT", "/home/irteam/AdsorbGen"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_WORKER = REPO / "scripts" / "replay" / "self_improve_lbfgs_worker.py"
_SPEC = importlib.util.spec_from_file_location("self_improve_lbfgs_worker", _WORKER)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load {_WORKER}")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
atomic_write_json = _MOD.atomic_write_json
build_unique_representatives = _MOD.build_unique_representatives


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-lmdb", nargs="+", required=True)
    p.add_argument("--gt-index", default="/home/irteam/data/replay/gt_index_by_sid_oc20_lbfgs.pkl")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20260523)
    p.add_argument("--num-systems", type=int, default=10000)
    p.add_argument("--num-placements", type=int, default=10)
    args = p.parse_args()

    with open(args.gt_index, "rb") as f:
        gt_index_by_sid = pickle.load(f)
    reps = build_unique_representatives(args.train_lmdb, gt_index_by_sid)
    if len(reps) < args.num_systems:
        raise RuntimeError(f"only {len(reps)} eligible unique systems, need {args.num_systems}")
    rng = np.random.default_rng(args.seed)
    selected_idx = sorted(rng.choice(len(reps), size=args.num_systems, replace=False).tolist())
    selected = [reps[i] for i in selected_idx]
    payload = {
        "seed": args.seed,
        "num_systems": args.num_systems,
        "num_placements": args.num_placements,
        "train_lmdb": args.train_lmdb,
        "num_eligible_unique_systems": len(reps),
        "systems": [
            {
                "system_key": list(r["system_key"]),
                "lmdb_id": int(r["lmdb_id"]),
                "raw_idx": int(r["raw_idx"]),
                "sid": int(r["sid"]),
                "E_sys_ref": float(r["E_sys_ref"]),
            }
            for r in selected
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, payload)
    print(json.dumps({k: payload[k] for k in payload if k != "systems"}, indent=2))


if __name__ == "__main__":
    main()
