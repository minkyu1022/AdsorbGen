#!/usr/bin/env python
"""Merge E_sys shards and rebuild replay gt_index on the UMA oc20 scale."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--old-gt-index", default="/home/irteam/data/replay/gt_index_by_sid.pkl")
    p.add_argument("--out-dir", default="/home/irteam/data/replay")
    p.add_argument("--require-all-shards", action="store_true")
    args = p.parse_args()

    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    e_sys = {}
    duplicate_sids = 0
    missing = []
    for shard in range(args.num_shards):
        path = shard_dir / f"e_sys_shard{shard}.pkl"
        if not path.exists():
            missing.append(str(path))
            continue
        with open(path, "rb") as f:
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
        raise RuntimeError("no E_sys shard records loaded")

    e_sys_path = out_dir / "E_sys.pkl"
    with open(e_sys_path, "wb") as f:
        pickle.dump(e_sys, f)
    print(f"Wrote merged E_sys: {len(e_sys)} sids -> {e_sys_path}")
    if duplicate_sids:
        print(f"WARNING: duplicate sids overwritten: {duplicate_sids}")
    if missing:
        print(f"WARNING: missing {len(missing)} shard file(s)")

    with open(args.old_gt_index, "rb") as f:
        old_gt = pickle.load(f)

    system_to_sids: dict[tuple, list[int]] = {}
    for sid, rec in e_sys.items():
        old = old_gt.get(int(sid))
        if not isinstance(old, dict) or old.get("system_key") is None:
            continue
        system_to_sids.setdefault(tuple(old["system_key"]), []).append(int(sid))

    gt_by_system = {}
    for system_key, sids in system_to_sids.items():
        valid = [
            (sid, float(e_sys[sid]["e_total"]))
            for sid in sids
            if e_sys[sid].get("e_total") is not None
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
        }

    gt_by_sid = {}
    for sid, rec in e_sys.items():
        old = old_gt.get(int(sid), {})
        system_key = tuple(old.get("system_key")) if old.get("system_key") is not None else None
        own = float(rec["e_total"]) if rec.get("e_total") is not None else None
        group = gt_by_system.get(system_key) if system_key is not None else None
        e_min = group["E_sys_min"] if group is not None else None
        e_mean = group["E_sys_mean"] if group is not None else None
        eligible = bool(
            rec.get("converged", False)
            and system_key is not None
            and e_mean is not None
        )
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
            "source": "E_sys.pkl/uma-s-1p1/oc20",
            "converged": bool(rec.get("converged", False)),
            "fmax": float(rec.get("fmax", float("nan"))),
            "n_steps": int(rec.get("n_steps", 0) or 0),
        }

    gt_sid_path = out_dir / "gt_index_by_sid_oc20.pkl"
    gt_sys_path = out_dir / "gt_index_by_system_oc20.pkl"
    with open(gt_sid_path, "wb") as f:
        pickle.dump(gt_by_sid, f)
    with open(gt_sys_path, "wb") as f:
        pickle.dump(gt_by_system, f)

    n_eligible = sum(1 for r in gt_by_sid.values() if r["eligible"])
    n_headroom = sum(
        1 for r in gt_by_sid.values()
        if r["eligible"] and r["improvement_headroom"] is not None and r["improvement_headroom"] > 0.05
    )
    n_mean_headroom = sum(
        1 for r in gt_by_sid.values()
        if r["eligible"] and r["mean_headroom"] is not None and r["mean_headroom"] > 0.0
    )
    print(f"Wrote oc20 gt_index_by_sid: {len(gt_by_sid)} sids, eligible={n_eligible} -> {gt_sid_path}")
    print(f"Wrote oc20 gt_index_by_system: {len(gt_by_system)} systems -> {gt_sys_path}")
    print(f"headroom > 0.05 eV: {n_headroom}")
    print(f"own energy above system mean: {n_mean_headroom}")


if __name__ == "__main__":
    main()
