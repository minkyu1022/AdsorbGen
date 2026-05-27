#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    log_dir = Path(args.out_dir) / "logs"
    rows = []
    for path in sorted(log_dir.glob("progress_shard*.json")):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        rows.append(d)
    if not rows:
        print(f"no progress files under {log_dir}")
        return
    candidates = sum(int(r.get("candidates", 0)) for r in rows)
    selected_path = Path(args.out_dir) / "selected_systems.json"
    if selected_path.exists():
        sel = json.loads(selected_path.read_text())
        target = int(sel.get("num_systems", 0)) * int(sel.get("num_placements", 0))
    else:
        target = sum(int(r.get("target_candidates", 0)) for r in rows)
    success = sum(int(r.get("success", 0)) for r in rows)
    valid = sum(int(r.get("valid", 0)) for r in rows)
    conv = sum(int(r.get("converged", 0)) for r in rows)
    # Systems are duplicated per shard progress denominators, so aggregate
    # systems-with-success is only exact after final merge. This is a lower
    # fidelity live view, but sample-level rates are exact.
    print(f"shards reporting: {len(rows)}")
    print(f"candidates: {candidates}/{target} ({candidates / max(target, 1) * 100:.2f}%)")
    print(f"converged: {conv}/{max(candidates,1)} ({conv / max(candidates, 1) * 100:.2f}%)")
    print(f"valid:     {valid}/{max(candidates,1)} ({valid / max(candidates, 1) * 100:.2f}%)")
    print(f"success:   {success}/{max(candidates,1)} ({success / max(candidates, 1) * 100:.4f}%)")
    for r in rows:
        print(
            f"  shard {r.get('shard_idx')}: "
            f"{r.get('candidates', 0)}/{r.get('target_candidates', 0)} "
            f"succ={r.get('success', 0)} "
            f"valid={r.get('valid_rate', 0.0) * 100:.1f}% "
            f"conv={r.get('converged_rate', 0.0) * 100:.1f}% "
            f"elapsed={float(r.get('elapsed_sec', 0.0)) / 60:.1f}m"
        )


if __name__ == "__main__":
    main()
