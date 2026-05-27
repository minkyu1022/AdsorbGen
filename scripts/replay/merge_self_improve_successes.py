#!/usr/bin/env python
"""Merge self-improvement shard successes and keep the best per system."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def frozen_key(x):
    if isinstance(x, (list, tuple)):
        return tuple(frozen_key(v) for v in x)
    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    successes = []
    for path in sorted(out_dir.glob("success_shard*.pkl")):
        with path.open("rb") as f:
            successes.extend(pickle.load(f))

    best = {}
    for row in successes:
        sk = frozen_key(row["system_key"])
        cur = best.get(sk)
        if cur is None or float(row["E_sys"]) < float(cur["E_sys"]):
            best[sk] = row

    best_rows = list(best.values())
    best_rows.sort(key=lambda r: (str(r["system_key"]), float(r["E_sys"])))
    with (out_dir / "replacement_best_by_system.pkl").open("wb") as f:
        pickle.dump(best_rows, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "success_entries": len(successes),
        "systems_with_replacement": len(best_rows),
        "output": str(out_dir / "replacement_best_by_system.pkl"),
    }
    (out_dir / "replacement_best_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
