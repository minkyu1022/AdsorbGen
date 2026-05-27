#!/usr/bin/env python3
from __future__ import annotations

import datetime as _dt
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(os.environ.get("ROOT", "/home/irteam"))
REPO = Path(os.environ.get("REPO", str(ROOT / "AdsorbGen")))
PY = os.environ.get("PY", "/home1/irteam/micromamba/envs/adsorbgen/bin/python")
REPLAY = Path(os.environ.get("REPLAY_1P2", str(ROOT / "data/replay_uma_s_1p2")))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "24"))
POLL_SEC = int(os.environ.get("POLL_SEC", "60"))
GPU_LIST = [int(x) for x in os.environ.get(
    "GPU_LIST_CSV",
    "0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7",
).split(",")]

LOG_DIR = REPLAY / "lbfgs_recompute_logs"
WATCH_LOG = REPLAY / "watch_eslab_manual_then_launch.py.log"
DONE_RE = re.compile(r"DONE\s+\d+\s+slabs\s+in\s+\d+s")


def log(msg: str) -> None:
    line = f"[watch-eslab-py] {_dt.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with WATCH_LOG.open("a") as f:
        f.write(line + "\n")


def shard_log(shard: int) -> Path:
    return LOG_DIR / f"e_slab_lbfgs_shard{shard}.manual.log"


def shard_pidfile(shard: int) -> Path:
    return LOG_DIR / f"e_slab_lbfgs_shard{shard}.manual.pid"


def is_done(shard: int) -> bool:
    path = shard_log(shard)
    if not path.exists():
        return False
    return bool(DONE_RE.search(path.read_text(errors="ignore")))


def read_pid(shard: int) -> int | None:
    path = shard_pidfile(shard)
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def launch_shard(shard: int) -> None:
    gpu = GPU_LIST[shard % len(GPU_LIST)]
    log(f"relaunch shard={shard} gpu={gpu}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = shard_log(shard).open("a")
    proc = subprocess.Popen(
        [
            PY,
            "scripts/replay/e_sys/compute_e_slab_lbfgs.py",
            "--shard-idx", str(shard),
            "--num-shards", str(NUM_SHARDS),
            "--out-dir", str(REPLAY / "e_slab_only_lbfgs_shards"),
            "--uma-model", "uma-s-1p2",
            "--uma-task", "oc20",
            "--uma-fmax", "0.05",
            "--uma-max-steps", "300",
            "--resume",
            "--pristine-slabs", str(ROOT / "results/pristine_slabs/is2res.pkl"),
        ],
        cwd=REPO,
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    shard_pidfile(shard).write_text(str(proc.pid))


def count_alive_compute() -> int:
    cmd = "pgrep -fc 'compute_e_slab_lbfgs.py' || true"
    out = subprocess.check_output(cmd, shell=True, text=True).strip()
    try:
        return int(out)
    except Exception:
        return 0


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().replace("\n", " | ")
    except Exception as exc:
        return f"nvidia-smi failed: {exc}"


def merge_and_launch() -> None:
    log("all shards done; merging E_slab")
    merge_log = LOG_DIR / "merge_e_slab_lbfgs.manual.log"
    with merge_log.open("w") as f:
        subprocess.check_call(
            [
                PY,
                "scripts/replay/e_sys/merge_e_slab_lbfgs.py",
                "--shard-dir", str(REPLAY / "e_slab_only_lbfgs_shards"),
                "--num-shards", str(NUM_SHARDS),
                "--sid-index", str(ROOT / "results/pristine_slabs/is2res.sid_index.pkl"),
                "--out-dir", str(REPLAY),
                "--uma-model", "uma-s-1p2",
                "--uma-task", "oc20",
                "--relaxed-pristine-out", str(REPLAY / "pristine_slabs_lbfgs.pkl"),
                "--require-all-shards",
            ],
            cwd=REPO,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    log("merge complete; launching post-reference watcher")
    post_log = REPLAY / "post_reference_watcher.manual_after_eslab.log"
    with post_log.open("w") as f:
        proc = subprocess.Popen(
            [
                "bash",
                "scripts/replay/wait_1p2_refs_then_launch_full_and_sde250.sh",
            ],
            cwd=REPO,
            env={**os.environ, "REPLAY_1P2": str(REPLAY), "POLL_SEC": "30"},
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (REPLAY / "post_reference_watcher.manual_after_eslab.pid").write_text(str(proc.pid))
    log(f"post-reference watcher pid={proc.pid}")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"started replay={REPLAY} poll={POLL_SEC}s shards={NUM_SHARDS}")
    while True:
        try:
            done = sum(is_done(i) for i in range(NUM_SHARDS))
            alive = count_alive_compute()
            log(f"tick alive={alive} done={done}/{NUM_SHARDS} gpu={gpu_snapshot()}")
            for shard in range(NUM_SHARDS):
                if is_done(shard):
                    continue
                pid = read_pid(shard)
                if not pid_alive(pid):
                    launch_shard(shard)
                    time.sleep(2)
            done = sum(is_done(i) for i in range(NUM_SHARDS))
            if done == NUM_SHARDS:
                merge_and_launch()
                return
        except Exception as exc:
            log(f"ERROR {type(exc).__name__}: {exc}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
