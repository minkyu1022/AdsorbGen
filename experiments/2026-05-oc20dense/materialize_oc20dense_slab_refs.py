#!/usr/bin/env python
"""Materialize OC20-Dense pristine slab MLIP references.

The pristine slab cache is keyed by unique slab geometry.  OC20-Dense replay
and global-minimum records are keyed by dense system_key, so this script writes
both views without recomputing relaxation.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def _slab_record(slab_key, rec):
    return {
        "slab_key": slab_key,
        "E_slab_only": float(rec["e_total"]),
        "e_total": float(rec["e_total"]),
        "converged": bool(rec.get("converged", False)),
        "fmax": float(rec.get("forces_max", float("nan"))),
        "n_steps": int(rec.get("n_steps", 0) or 0),
        "n_atoms": int(rec.get("n_atoms", 0) or 0),
        "pos_relaxed": rec.get("pos"),
        "atomic_numbers": rec.get("atomic_numbers"),
        "tags": rec.get("tags"),
        "fixed": rec.get("fixed"),
        "cell": rec.get("cell"),
        "pbc": rec.get("pbc"),
        "source": "uma-s-1p1/oc20/pristine_slab",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system-index", default="/home/irteam/results/pristine_slabs/oc20dense.system_index.pkl")
    ap.add_argument("--pristine-slabs", default="/home/irteam/results/pristine_slabs/oc20dense_uma.pkl")
    ap.add_argument("--out-dir", default="/home/irteam/data/replay")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.system_index, "rb") as f:
        dense_system_to_slab = pickle.load(f)
    with open(args.pristine_slabs, "rb") as f:
        slab_cache = pickle.load(f)

    unique_slabs = set(dense_system_to_slab.values())
    missing_slabs = sorted(unique_slabs - set(slab_cache.keys()), key=str)

    by_slab = {}
    for slab_key in sorted(unique_slabs, key=str):
        rec = slab_cache.get(slab_key)
        if rec is None or rec.get("e_total") is None:
            continue
        by_slab[slab_key] = _slab_record(slab_key, rec)

    by_system = {}
    missing_systems = []
    for system_key, slab_key in dense_system_to_slab.items():
        slab_rec = by_slab.get(slab_key)
        if slab_rec is None:
            missing_systems.append(system_key)
            continue
        out = dict(slab_rec)
        out["system_key"] = str(system_key)
        by_system[str(system_key)] = out

    unconverged_slabs = [
        str(k) for k, rec in by_slab.items() if not rec.get("converged", False)
    ]
    summary = {
        "source_pristine_slabs": str(args.pristine_slabs),
        "source_system_index": str(args.system_index),
        "dense_systems": len(dense_system_to_slab),
        "unique_slabs": len(unique_slabs),
        "slab_records_written": len(by_slab),
        "system_records_written": len(by_system),
        "missing_slabs": len(missing_slabs),
        "missing_systems": len(missing_systems),
        "unconverged_slabs": len(unconverged_slabs),
        "unconverged_slab_examples": unconverged_slabs[:10],
        "by_slab": str(out_dir / "oc20dense_E_slab_only_by_slab.pkl"),
        "by_system": str(out_dir / "oc20dense_E_slab_only_by_system.pkl"),
    }

    with (out_dir / "oc20dense_E_slab_only_by_slab.pkl").open("wb") as f:
        pickle.dump(by_slab, f)
    with (out_dir / "oc20dense_E_slab_only_by_system.pkl").open("wb") as f:
        pickle.dump(by_system, f)
    (out_dir / "oc20dense_E_slab_only_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
