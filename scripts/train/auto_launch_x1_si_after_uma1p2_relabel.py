#!/usr/bin/env python
"""Launch two x1 SI scratch runs after UMA-s-1p2 relabeled data is complete."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home1/irteam")
REPO = ROOT / "AdsorbGen"
BASE = ROOT / "data/uma_s_1p2_references"
RUN_ROOT = ROOT / "runs/training"
LOG = BASE / "auto_launch_x1_si_after_uma1p2_relabel.log"
LAUNCH = REPO / "scripts/train/launch_x1_si_uma1p2_relabel_4gpu.sh"

READY_FILES = [
    BASE / "processed/oc20dense_unwrap_centered.report.json",
    BASE / "processed/is2res_train_unwrap_centered.report.json",
    BASE / "processed/is2res_val_unwrap_centered.report.json",
    BASE / "materialized/oc20dense_raw/oc20dense_mlip_global_min_by_system.pkl",
    BASE / "materialized/bare_slab/E_bare_slab_lbfgs_summary.json",
]


def log(msg: str) -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def run_exists(run_name: str) -> bool:
    out = RUN_ROOT / run_name
    return out.exists() and any(out.iterdir())


def train_running(run_name: str) -> bool:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    except Exception:
        return False
    needle = str(RUN_ROOT / run_name)
    return any("adsorbgen.training.train_cli" in line and needle in line for line in out.splitlines())


def launch_one(run_name: str, cuda: str, port: int, sampling: str,
               alpha: float = 2.0, beta: float = 1.0) -> None:
    out = RUN_ROOT / run_name
    if run_exists(run_name):
        log(f"output dir already has files; skip scratch launch for {run_name}")
        return
    out.mkdir(parents=True, exist_ok=True)
    if train_running(run_name):
        log(f"already running {run_name}")
        return
    marker = out / ".auto_launch_started"
    if marker.exists():
        log(f"launch marker exists; skip {run_name}")
        return
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": cuda,
        "MASTER_PORT": str(port),
        "RUN_NAME": run_name,
        "OUT": str(out),
        "TRAIN_TIME_SAMPLING": sampling,
        "TRAIN_TIME_BETA_ALPHA": str(alpha),
        "TRAIN_TIME_BETA_BETA": str(beta),
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "online",
    })
    log_path = out / "train.log"
    with log_path.open("w") as f:
        subprocess.Popen(
            ["setsid", "-f", "bash", str(LAUNCH)],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    marker.write_text(time.strftime("%F %T") + "\n")
    log(f"launched {run_name} cuda={cuda} port={port} sampling={sampling}")


def main() -> None:
    log("x1 SI auto-launch watcher start")
    while True:
        missing = [p for p in READY_FILES if not p.exists()]
        if not missing:
            break
        log("waiting for relabel outputs: " + ", ".join(str(p) for p in missing[:3]))
        time.sleep(300)

    uniform = "x1_SI_vloss_eta_102M_sigma0p1_w0p5_uma1p2_uniform"
    beta = "x1_SI_vloss_eta_102M_sigma0p1_w0p5_uma1p2_beta2_1"
    if run_exists(uniform) or run_exists(beta):
        log(
            "one or more output dirs already contain files; not deleting or overwriting. "
            f"uniform_exists={run_exists(uniform)} beta_exists={run_exists(beta)}"
        )
    launch_one(uniform, "0,1,2,3", 29661, "uniform")
    launch_one(beta, "4,5,6,7", 29662, "beta", 2.0, 1.0)
    log("x1 SI auto-launch watcher done")


if __name__ == "__main__":
    main()
