#!/usr/bin/env python
"""Continue UMA-s-1p2 relabeling without leaving GPUs idle.

Current scope:
  1. keep processed_ID train shards running until all complete;
  2. run processed_ID val shards;
  3. materialize train/val LMDBs from both shard dirs;
  4. unwrap/center the materialized LMDBs for training use.
"""

from __future__ import annotations

import os
import pickle
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path("/home1/irteam")
REPO = ROOT / "AdsorbGen"
PY = ROOT / "micromamba/envs/adsorbgen/bin/python"
BASE = ROOT / "data/uma_s_1p2_references"
LOG = BASE / "auto_continue_id_uma1p2.log"

TRAIN_RAW = BASE / "raw_relax/processed_ID_train"
VAL_RAW = BASE / "raw_relax/processed_ID_val"
MAT = BASE / "materialized/processed_ID_raw"
PROC = BASE / "processed"

COMMON = [
    "--source-kind", "id_lmdb",
    "--num-shards", "8",
    "--records-per-chunk", "512",
    "--skip-anomaly-mask",
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


def log(msg: str) -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def ps_lines() -> list[str]:
    out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    return out.splitlines()


def shard_running(out_dir: Path, shard: int) -> bool:
    needle = str(out_dir)
    shard_arg = f"--shard-idx {shard}"
    for line in ps_lines():
        if (
            "batched_uma_relabel.py" in line
            and "--source-kind id_lmdb" in line
            and needle in line
            and shard_arg in line
        ):
            return True
    return False


def shard_done(out_dir: Path, shard: int) -> bool:
    return (out_dir / f"id_lmdb_shard{shard}.summary.pkl").is_file()


def free_gpus() -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    free = []
    for line in out.splitlines():
        idx_s, mem_s = [x.strip() for x in line.split(",")[:2]]
        if int(mem_s) < 2048:
            free.append(int(idx_s))
    return free


def launch_shard(out_dir: Path, lmdb: Path, shard: int, gpu: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    log_path = out_dir / "logs" / f"relabel_{out_dir.name}_shard{shard}.log"
    cmd = [
        "setsid", "-f", "env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        f"PYTHONPATH={REPO}",
        str(PY), "-u", str(REPO / "scripts/replay/batched_uma_relabel.py"),
        "--input", str(lmdb),
        "--out-dir", str(out_dir),
        "--shard-idx", str(shard),
        *COMMON,
    ]
    with log_path.open("w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    log(f"launched {out_dir.name} shard={shard} gpu={gpu}")


def ensure_stage(out_dir: Path, lmdb: Path) -> bool:
    for gpu in free_gpus():
        missing = [
            s for s in range(8)
            if not shard_done(out_dir, s) and not shard_running(out_dir, s)
        ]
        if not missing:
            break
        launch_shard(out_dir, lmdb, missing[0], gpu)
        time.sleep(3)
    return all(shard_done(out_dir, s) for s in range(8))


def run_once(cmd: list[str], log_path: Path, marker: Path) -> None:
    if marker.exists():
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"run {' '.join(shlex.quote(x) for x in cmd)}")
    with log_path.open("w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)


def materialize_and_preprocess() -> None:
    MAT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    summary = MAT / "build_summary.json"
    run_once(
        [
            str(PY), str(REPO / "scripts/replay/materialize_processed_id_uma1p2.py"),
            "--shard-dirs", str(TRAIN_RAW), str(VAL_RAW),
            "--num-shards", "8",
            "--out-dir", str(MAT),
            "--lmdbs",
            str(ROOT / "data/processed_ID/is2res_train.lmdb"),
            str(ROOT / "data/processed_ID/is2res_val.lmdb"),
            "--require-all-shards",
        ],
        MAT / "materialize_processed_ID.log",
        summary,
    )
    for name in ("is2res_train", "is2res_val"):
        src = MAT / f"{name}.lmdb"
        dst = PROC / f"{name}_unwrap_centered.lmdb"
        marker = dst.with_suffix(".report.json")
        run_once(
            [
                str(PY), "-m", "adsorbgen.scripts.unwrap_preprocess",
                "--src", str(src),
                "--dst", str(dst),
                "--adsorbates-pkl", str(REPO / "data/pkls/adsorbates.pkl"),
                "--center-mode", "relaxed_all",
                "--pbc-axes", "xy",
            ],
            MAT / f"unwrap_center_{name}.log",
            marker,
        )


def main() -> None:
    log("auto watcher start")
    train_lmdb = ROOT / "data/processed_ID/is2res_train.lmdb"
    val_lmdb = ROOT / "data/processed_ID/is2res_val.lmdb"
    while True:
        train_done = ensure_stage(TRAIN_RAW, train_lmdb)
        if not train_done:
            time.sleep(60)
            continue
        log("train relabel complete")
        val_done = ensure_stage(VAL_RAW, val_lmdb)
        if not val_done:
            time.sleep(60)
            continue
        log("val relabel complete")
        materialize_and_preprocess()
        log("processed_ID materialize/preprocess complete")
        return


if __name__ == "__main__":
    main()
