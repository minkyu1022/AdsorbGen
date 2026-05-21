#!/usr/bin/env python
"""Monitor OC20-Dense UMA relaxation shards and print ETA."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


PROG_RE = re.compile(
    r"(?P<done>\d+)/(?P<total>\d+)\s+\[(?P<elapsed>\d+:\d+(?::\d+)?)[<,].*?"
    r"conv=(?P<conv>\d+).*?records=(?P<records>\d+)"
)


def parse_elapsed(value: str) -> int:
    parts = [int(x) for x in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    matches = list(PROG_RE.finditer(text.replace("\r", "\n")))
    if not matches:
        assigned = None
        m = re.search(r"assigned\s+(\d+)\s+/", text)
        if m:
            assigned = int(m.group(1))
        return {
            "done": 0,
            "total": None,
            "records": 0,
            "conv": 0,
            "assigned": assigned,
            "elapsed_sec": 0,
        }
    m = matches[-1]
    return {
        "done": int(m.group("done")),
        "total": int(m.group("total")),
        "records": int(m.group("records")),
        "conv": int(m.group("conv")),
        "assigned": None,
        "elapsed_sec": parse_elapsed(m.group("elapsed")),
    }


def proc_count() -> int:
    out = subprocess.run(
        ["pgrep", "-f", "compute_oc20dense_mlip_relax.py --lmdb /home/irteam/data/processed/oc20dense.lmdb"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.strip()
    return len([x for x in out.splitlines() if x.strip()])


def gpu_line() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return " | ".join(line.strip() for line in out.splitlines())
    except Exception as exc:  # pragma: no cover
        return f"nvidia-smi failed: {exc}"


def summarize(log_dir: Path, summary_path: Path) -> dict:
    rows = []
    for p in sorted(log_dir.glob("shard_*.log")):
        row = parse_log(p)
        row["shard"] = p.stem.split("_")[-1]
        rows.append(row)

    now = datetime.now()
    total_done = sum(r["done"] for r in rows)
    total_chunks = sum(r["total"] or 0 for r in rows)
    total_records = sum(r["records"] for r in rows)
    total_conv = sum(r["conv"] for r in rows)
    eta = None
    shard_remaining = []
    for r in rows:
        if r["done"] > 0 and r["total"]:
            sec_per_chunk = r["elapsed_sec"] / max(r["done"], 1)
            shard_remaining.append(sec_per_chunk * max(r["total"] - r["done"], 0))
    if shard_remaining:
        eta = now + timedelta(seconds=max(shard_remaining))

    return {
        "time": now.isoformat(timespec="seconds"),
        "active_relax_processes": proc_count(),
        "gpu": gpu_line(),
        "chunks_done": total_done,
        "chunks_total": total_chunks,
        "records_done": total_records,
        "converged_done": total_conv,
        "eta": eta.isoformat(timespec="seconds") if eta else None,
        "merge_summary_exists": summary_path.exists(),
        "shards": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="/home/irteam/data/replay/oc20dense_mlip_relax_logs")
    ap.add_argument("--summary", default="/home/irteam/data/replay/oc20dense_mlip_relax_summary.json")
    ap.add_argument("--interval-sec", type=int, default=600)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    summary_path = Path(args.summary)
    while True:
        s = summarize(log_dir, summary_path)
        print(json.dumps(s, sort_keys=True), flush=True)
        if args.once or s["merge_summary_exists"]:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
