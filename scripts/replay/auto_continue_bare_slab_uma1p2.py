#!/usr/bin/env python
"""Start bare-slab UMA-s-1p2 relabeling after processed_ID relabel finishes."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


ROOT = Path("/home1/irteam")
REPO = ROOT / "AdsorbGen"
PY = ROOT / "micromamba/envs/adsorbgen/bin/python"
BASE = ROOT / "data/uma_s_1p2_references"
LOG = BASE / "auto_continue_bare_slab_uma1p2.log"
BARE_RAW = BASE / "raw_relax/bare_slab"
BARE_MAT = BASE / "materialized/bare_slab"
INPUT = ROOT / "data-vol1/minkyu/data/replay/E_slab_only_lbfgs_by_slab.pkl"


def log(msg: str) -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def free_gpus() -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    free = []
    for line in out.splitlines():
        idx, mem = [x.strip() for x in line.split(",")[:2]]
        if int(mem) < 2048:
            free.append(int(idx))
    return free


def ps_text() -> str:
    return subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)


def running(shard: int) -> bool:
    for line in ps_text().splitlines():
        if (
            "batched_uma_relabel.py" in line
            and "--source-kind bare_slab_pkl" in line
            and str(BARE_RAW) in line
            and f"--shard-idx {shard}" in line
        ):
            return True
    return False


def done(shard: int) -> bool:
    return (BARE_RAW / f"bare_slab_pkl_shard{shard}.summary.pkl").is_file()


def launch(shard: int, gpu: int) -> None:
    BARE_RAW.mkdir(parents=True, exist_ok=True)
    (BARE_RAW / "logs").mkdir(exist_ok=True)
    log_path = BARE_RAW / "logs" / f"relabel_bare_slab_shard{shard}.log"
    cmd = [
        "setsid", "-f", "env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        f"PYTHONPATH={REPO}",
        str(PY), "-u", str(REPO / "scripts/replay/batched_uma_relabel.py"),
        "--source-kind", "bare_slab_pkl",
        "--input", str(INPUT),
        "--out-dir", str(BARE_RAW),
        "--shard-idx", str(shard),
        "--num-shards", "8",
        "--records-per-chunk", "512",
        "--uma-model", "uma-s-1p2",
        "--uma-task", "oc20",
        "--fmax", "0.05",
        "--max-steps", "300",
        "--max-atoms", "65536",
        "--maxstep", "0.2",
        "--lbfgs-memory", "100",
        "--lbfgs-streaming",
        "--lbfgs-check-interval", "20",
        "--lbfgs-keep-survivors-on-gpu",
        "--resume",
    ]
    with log_path.open("w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    log(f"launched bare slab shard={shard} gpu={gpu}")


def main() -> None:
    log("bare slab watcher start")
    train_report = BASE / "processed/is2res_train_unwrap_centered.report.json"
    val_report = BASE / "processed/is2res_val_unwrap_centered.report.json"
    while not (train_report.exists() and val_report.exists()):
        time.sleep(120)
    log("processed_ID preprocess complete; start/monitor bare slab")
    while True:
        for gpu in free_gpus():
            missing = [s for s in range(8) if not done(s) and not running(s)]
            if not missing:
                break
            launch(missing[0], gpu)
            time.sleep(3)
        if all(done(s) for s in range(8)):
            break
        time.sleep(60)
    log("bare slab relabel complete; materialize")
    BARE_MAT.mkdir(parents=True, exist_ok=True)
    with (BARE_MAT / "materialize_bare_slab.log").open("w") as f:
        subprocess.run(
            [
                str(PY), str(REPO / "scripts/replay/materialize_bare_slab_uma1p2.py"),
                "--shard-dir", str(BARE_RAW),
                "--num-shards", "8",
                "--out-dir", str(BARE_MAT),
                "--require-all-shards",
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    log("bare slab materialize complete")


if __name__ == "__main__":
    main()
