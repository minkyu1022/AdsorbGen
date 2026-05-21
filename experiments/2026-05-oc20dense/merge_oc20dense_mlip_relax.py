#!/usr/bin/env python
"""Merge OC20-Dense UMA relax shards and build per-system MLIP minima."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", default="/home/irteam/data/replay/oc20dense_mlip_relax_shards")
    ap.add_argument("--out-dir", default="/home/irteam/data/replay")
    ap.add_argument("--num-shards", type=int, default=4)
    args = ap.parse_args()

    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    for shard in range(args.num_shards):
        p = shards_dir / f"oc20dense_mlip_relax_shard{shard}.pkl"
        if not p.exists():
            raise FileNotFoundError(p)
        with p.open("rb") as f:
            part = pickle.load(f)
        records.update(part)

    by_system = defaultdict(list)
    for idx, rec in records.items():
        by_system[str(rec["system_key"])].append((int(idx), rec))

    minima = {}
    for system_key, items in by_system.items():
        converged = [(idx, rec) for idx, rec in items if rec.get("converged")]
        pool = converged if converged else items
        idx_min, rec_min = min(pool, key=lambda x: float(x[1]["e_total"]))
        e_vals = [float(rec["e_total"]) for _, rec in converged]
        minima[system_key] = {
            "system_key": system_key,
            "idx_min": int(idx_min),
            "config_key_min": rec_min.get("config_key"),
            "E_sys_min": float(rec_min["e_total"]),
            "E_sys_mean_converged": float(sum(e_vals) / len(e_vals)) if e_vals else None,
            "n_configs": len(items),
            "n_converged": len(converged),
            "all_converged": len(converged) == len(items),
            "used_unconverged_min": not bool(converged),
            "fmax_min": float(rec_min.get("fmax", float("nan"))),
            "n_steps_min": int(rec_min.get("n_steps", -1)),
            "pos_relaxed": rec_min["pos_relaxed"],
            "source": "uma-s-1p1/oc20/oc20dense_mlip_relax",
        }

    full_path = out_dir / "oc20dense_mlip_relax.pkl"
    min_path = out_dir / "oc20dense_mlip_global_min_by_system.pkl"
    summary_path = out_dir / "oc20dense_mlip_relax_summary.json"
    with full_path.open("wb") as f:
        pickle.dump(records, f)
    with min_path.open("wb") as f:
        pickle.dump(minima, f)

    n_conv = sum(1 for r in records.values() if r.get("converged"))
    summary = {
        "source": "uma-s-1p1/oc20",
        "records": len(records),
        "systems": len(minima),
        "converged_records": n_conv,
        "converged_rate": n_conv / max(len(records), 1),
        "systems_all_converged": sum(1 for r in minima.values() if r["all_converged"]),
        "systems_with_no_converged_config": sum(1 for r in minima.values() if r["used_unconverged_min"]),
        "full_records": str(full_path),
        "global_min_by_system": str(min_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
