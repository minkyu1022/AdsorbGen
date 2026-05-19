#!/usr/bin/env python
"""Aggregate one replay-daemon cycle across shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SUM_KEYS = [
    "n_systems",
    "candidates",
    "n_success",
    "n_success_systems",
    "n_added",
    "n_streamed",
    "n_relaxed_success_plus_0p1",
    "n_relaxed_success_plus_0p2",
    "n_relaxed_success_plus_0p3",
    "n_relaxed_success_systems_plus_0p1",
    "n_relaxed_success_systems_plus_0p2",
    "n_relaxed_success_systems_plus_0p3",
]

RATE_KEYS = [
    "valid_rate",
    "dissoc_rate",
    "desorbed_rate",
    "surf_changed_rate",
    "intercalated_rate",
    "overlap_rate",
    "uma_unconverged_rate",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stream-dir", required=True)
    p.add_argument("--cycle", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    stream_dir = Path(args.stream_dir)
    files = sorted((stream_dir / "logs").glob(f"cycle_{args.cycle:06d}_shard*.json"))
    if not files:
        raise SystemExit(f"no cycle files found under {stream_dir}/logs")

    rows = [json.loads(path.read_text()) for path in files]
    total = {k: 0 for k in SUM_KEYS}
    for row in rows:
        for k in SUM_KEYS:
            total[k] += int(row.get(k, 0))

    candidates = max(int(total["candidates"]), 1)
    systems = max(int(total["n_systems"]), 1)
    report = {
        "stream_dir": str(stream_dir),
        "cycle": args.cycle,
        "num_shards": len(rows),
        **total,
        "strict_success_candidate_rate": total["n_success"] / candidates,
        "strict_success_system_rate": total["n_success_systems"] / systems,
        "relaxed_plus_0p1_candidate_rate": total["n_relaxed_success_plus_0p1"] / candidates,
        "relaxed_plus_0p2_candidate_rate": total["n_relaxed_success_plus_0p2"] / candidates,
        "relaxed_plus_0p3_candidate_rate": total["n_relaxed_success_plus_0p3"] / candidates,
        "relaxed_plus_0p1_system_rate": total["n_relaxed_success_systems_plus_0p1"] / systems,
        "relaxed_plus_0p2_system_rate": total["n_relaxed_success_systems_plus_0p2"] / systems,
        "relaxed_plus_0p3_system_rate": total["n_relaxed_success_systems_plus_0p3"] / systems,
        "elapsed_sec_max": max(float(r.get("elapsed_sec", 0.0)) for r in rows),
        "gpu_util_mean_by_shard": [
            r.get("gpu_util", {}).get("mean") for r in rows
        ],
    }

    for k in RATE_KEYS:
        weighted = 0.0
        for row in rows:
            weighted += float(row.get(k, 0.0)) * int(row.get("candidates", 0))
        report[k] = weighted / candidates

    out = Path(args.out) if args.out else stream_dir / f"cycle_{args.cycle:06d}_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()
