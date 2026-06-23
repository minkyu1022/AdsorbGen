#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def load_eval_module(path: Path):
    spec = importlib.util.spec_from_file_location("goal1_eval_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-script", required=True)
    ap.add_argument("--shard-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()

    eval_mod = load_eval_module(Path(args.eval_script))
    rows = []
    pred_relax = []
    target_relax = []
    settings = None
    for shard_idx in range(int(args.num_shards)):
        path = Path(args.shard_root) / f"shard{shard_idx}" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        settings = settings or payload.get("settings", {})
        rows.extend(payload["rows"])
        pred_relax.extend(payload["pred_relax"])
        target_relax.extend(payload["target_relax"])

    rows.sort(key=lambda r: int(r["global_i"]))
    pred_relax.sort(key=lambda r: int(r["global_i"]))
    target_relax.sort(key=lambda r: int(r["global_i"]))
    payload = {
        "settings": settings or {},
        "summary": eval_mod.summarize(rows, pred_relax, target_relax),
        "rows": rows,
        "pred_relax": pred_relax,
        "target_relax": target_relax,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, out_dir / "summary.json")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
