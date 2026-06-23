#!/usr/bin/env python
"""Monitor MLIP pass@k shards and send a Slack completion notification."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path


PROGRESS_RE = re.compile(r"flow\+lbfgs:\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)")


def send_slack(webhook: str, text: str) -> None:
    data = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        resp.read()


def tail_text(path: Path, n_bytes: int = 8192) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n_bytes), os.SEEK_SET)
        return f.read().decode("utf-8", errors="replace")


def shard_progress(log_dir: Path, num_shards: int) -> dict[int, tuple[int, int]]:
    out = {}
    for shard in range(num_shards):
        text = tail_text(log_dir / f"shard_{shard}.log")
        matches = PROGRESS_RE.findall(text)
        if matches:
            _, done, total = matches[-1]
            out[shard] = (int(done), int(total))
        else:
            out[shard] = (0, 20)
    return out


def is_supervisor_alive(pid_file: Path) -> bool:
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        return False
    return Path(f"/proc/{pid}").exists()


def format_summary(summary: dict, out_dir: Path, elapsed_min: float) -> str:
    def f(key: str) -> str:
        v = summary.get(key)
        return "NA" if v is None else f"{float(v):.4f}"

    return (
        "*MLIP pass@k finished*\n"
        f"out: `{out_dir}`\n"
        f"elapsed: {elapsed_min:.1f} min\n"
        f"systems: {summary.get('systems')}  candidates: {summary.get('candidates')}\n"
        f"converged_rate: {f('converged_rate')}\n"
        f"valid_rate: {f('valid_rate')}\n"
        f"success_sample_rate: {f('success_sample_rate')}\n"
        f"pass@1: {f('mlip_pass@1')}\n"
        f"pass@2: {f('mlip_pass@2')}\n"
        f"pass@5: {f('mlip_pass@5')}\n"
        f"pass@10: {f('mlip_pass@10')}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--poll-sec", type=int, default=300)
    ap.add_argument("--webhook", default=os.environ.get("SLACK_WEBHOOK_URL", ""))
    ap.add_argument("--notify-start", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    log_dir = out_dir / "logs"
    monitor_log = log_dir / "slack_monitor.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.webhook:
        raise SystemExit("missing Slack webhook: pass --webhook or set SLACK_WEBHOOK_URL")

    t0 = time.time()
    if args.notify_start:
        send_slack(args.webhook, f"MLIP pass@k monitor attached: `{out_dir}`")

    while True:
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            text = format_summary(summary, out_dir, (time.time() - t0) / 60.0)
            send_slack(args.webhook, text)
            (out_dir / "SLACK_NOTIFIED").write_text(str(time.time()))
            with monitor_log.open("a") as f:
                f.write(f"[done] notified at {time.ctime()}\n")
            return

        progress = shard_progress(log_dir, args.num_shards)
        alive = is_supervisor_alive(log_dir / "supervisor.pid")
        with monitor_log.open("a") as f:
            compact = " ".join(f"s{s}:{d}/{t}" for s, (d, t) in progress.items())
            f.write(f"[poll] {time.ctime()} alive={alive} {compact}\n")

        failed = (not alive) and not summary_path.exists()
        if failed:
            try:
                tail = subprocess.check_output(
                    ["tail", "-60", str(log_dir / "supervisor.log")],
                    text=True,
                    stderr=subprocess.STDOUT,
                )
            except Exception as exc:
                tail = repr(exc)
            send_slack(
                args.webhook,
                "*MLIP pass@k monitor: supervisor stopped without summary*\n"
                f"out: `{out_dir}`\n"
                f"supervisor tail:\n```{tail[-2500:]}```",
            )
            raise SystemExit(1)

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
