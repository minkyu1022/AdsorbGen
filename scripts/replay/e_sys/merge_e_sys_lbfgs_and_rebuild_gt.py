#!/usr/bin/env python
"""Merge L-BFGS E_sys shards and rebuild OC20-scale replay indices."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def _frozen_key(x):
    if isinstance(x, (list, tuple)):
        return tuple(_frozen_key(v) for v in x)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=24)
    ap.add_argument("--old-gt-index", default="/home/irteam/data/replay/gt_index_by_sid.pkl")
    ap.add_argument("--out-dir", default="/home/irteam/data/replay")
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--require-all-shards", action="store_true")
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    e_sys: dict[int, dict] = {}
    missing = []
    duplicate_sids = 0
    for shard in range(args.num_shards):
        path = shard_dir / f"e_sys_lbfgs_shard{shard}.pkl"
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open("rb") as f:
            part = pickle.load(f)
        for sid, rec in part.items():
            sid = int(sid)
            if sid in e_sys:
                duplicate_sids += 1
            e_sys[sid] = rec
        print(f"loaded shard {shard}: {len(part)} records")

    if missing and args.require_all_shards:
        raise FileNotFoundError("missing shard files:\n" + "\n".join(missing))
    if not e_sys:
        raise RuntimeError("no E_sys L-BFGS shard records loaded")

    with open(args.old_gt_index, "rb") as f:
        old_gt = pickle.load(f)

    system_to_sids: dict[tuple, list[int]] = {}
    for sid, rec in e_sys.items():
        old = old_gt.get(int(sid))
        if not isinstance(old, dict) or old.get("system_key") is None:
            continue
        system_to_sids.setdefault(_frozen_key(old["system_key"]), []).append(int(sid))

    gt_by_system = {}
    for system_key, sids in system_to_sids.items():
        valid = [
            (sid, float(e_sys[sid]["e_total"]))
            for sid in sids
            if e_sys[sid].get("converged") and e_sys[sid].get("e_total") is not None
        ]
        if not valid:
            continue
        sid_min, e_min = min(valid, key=lambda x: x[1])
        e_mean = sum(e for _, e in valid) / len(valid)
        gt_by_system[system_key] = {
            "E_sys_min": float(e_min),
            "E_sys_mean": float(e_mean),
            "sid_min": int(sid_min),
            "sids": sorted(int(sid) for sid, _ in valid),
            "n_sids": len(valid),
            "source": f"E_sys_lbfgs.pkl/{args.uma_model}/{args.uma_task}/ase.LBFGS",
        }

    gt_by_sid = {}
    for sid, rec in e_sys.items():
        old = old_gt.get(int(sid), {})
        system_key = _frozen_key(old.get("system_key")) if old.get("system_key") is not None else None
        own = float(rec["e_total"]) if rec.get("e_total") is not None else None
        group = gt_by_system.get(system_key) if system_key is not None else None
        e_min = group["E_sys_min"] if group is not None else None
        e_mean = group["E_sys_mean"] if group is not None else None
        eligible = bool(rec.get("converged", False) and system_key is not None and e_mean is not None)
        gt_by_sid[int(sid)] = {
            "system_key": system_key,
            "E_sys_min": float(e_min) if e_min is not None else None,
            "E_sys_mean": float(e_mean) if e_mean is not None else None,
            "E_sys_own": own,
            "eligible": eligible,
            "improvement_headroom": (
                float(own - e_min) if eligible and own is not None and e_min is not None else None
            ),
            "mean_headroom": (
                float(own - e_mean) if eligible and own is not None and e_mean is not None else None
            ),
            "source": f"E_sys_lbfgs.pkl/{args.uma_model}/{args.uma_task}/ase.LBFGS",
            "converged": bool(rec.get("converged", False)),
            "fmax": float(rec.get("fmax", float("nan"))),
            "n_steps": int(rec.get("n_steps", 0) or 0),
        }

    e_sys_path = out_dir / "E_sys_lbfgs.pkl"
    gt_sid_path = out_dir / "gt_index_by_sid_oc20_lbfgs.pkl"
    gt_sys_path = out_dir / "gt_index_by_system_oc20_lbfgs.pkl"
    summary_path = out_dir / "E_sys_lbfgs_summary.json"
    with e_sys_path.open("wb") as f:
        pickle.dump(e_sys, f, protocol=pickle.HIGHEST_PROTOCOL)
    with gt_sid_path.open("wb") as f:
        pickle.dump(gt_by_sid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with gt_sys_path.open("wb") as f:
        pickle.dump(gt_by_system, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "source": f"{args.uma_model}/{args.uma_task}/ase.LBFGS",
        "uma_model": args.uma_model,
        "uma_task": args.uma_task,
        "relaxer": "ase.LBFGS",
        "records": len(e_sys),
        "converged_records": sum(1 for r in e_sys.values() if r.get("converged")),
        "gt_sid_records": len(gt_by_sid),
        "gt_system_records": len(gt_by_system),
        "duplicate_sids": duplicate_sids,
        "missing_shards": missing,
        "E_sys_lbfgs": str(e_sys_path),
        "gt_index_by_sid_oc20_lbfgs": str(gt_sid_path),
        "gt_index_by_system_oc20_lbfgs": str(gt_sys_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
