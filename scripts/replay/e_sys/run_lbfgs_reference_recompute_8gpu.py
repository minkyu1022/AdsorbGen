#!/usr/bin/env python
"""Supervise L-BFGS reference recomputation across 8 GPUs.

Runs stages sequentially:
1. E_sys L-BFGS shards
2. merge E_sys_lbfgs + rebuild gt_index *_lbfgs
3. E_slab_only L-BFGS shards
4. merge E_slab_only_lbfgs
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(os.environ.get("ROOT", "/home/irteam"))
REPO = Path(os.environ.get("REPO", str(ROOT / "AdsorbGen")))
PY = os.environ.get("PY", "/home1/irteam/micromamba/envs/adsorbgen/bin/python")
REPLAY_DIR = Path(os.environ.get("REPLAY_DIR", str(ROOT / "data" / "replay")))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "24"))
GPU_LIST = os.environ.get(
    "GPU_LIST",
    "0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7",
).split()
UMA_MODEL = os.environ.get("UMA_MODEL", "uma-s-1p1")
UMA_TASK = os.environ.get("UMA_TASK", "oc20")
UMA_FMAX = os.environ.get("UMA_FMAX", "0.05")
UMA_MAX_STEPS = os.environ.get("UMA_MAX_STEPS", "300")
OLD_GT_INDEX = Path(os.environ.get("OLD_GT_INDEX", str(ROOT / "data" / "replay" / "gt_index_by_sid.pkl")))

E_SYS_SHARDS = Path(os.environ.get("E_SYS_SHARDS", str(REPLAY_DIR / "e_sys_lbfgs_shards")))
E_SLAB_SHARDS = Path(os.environ.get("E_SLAB_SHARDS", str(REPLAY_DIR / "e_slab_only_lbfgs_shards")))
LOG_DIR = Path(os.environ.get("LOG_DIR", str(REPLAY_DIR / "lbfgs_recompute_logs")))


def _complete_shards(out_dir: Path, prefix: str) -> bool:
    return all((out_dir / f"{prefix}_shard{shard}.pkl").is_file() for shard in range(NUM_SHARDS))


def _run_shards(stage: str, script: str, out_dir: Path, log_prefix: str, extra_args: list[str]) -> None:
    print(f"[lbfgs] {stage}: launching {NUM_SHARDS} shards", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[int, subprocess.Popen]] = []
    for shard in range(NUM_SHARDS):
        gpu = GPU_LIST[shard % len(GPU_LIST)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = LOG_DIR / f"{log_prefix}_shard{shard}.log"
        pid_path = LOG_DIR / f"{log_prefix}_shard{shard}.pid"
        cmd = [
            PY,
            script,
            "--shard-idx",
            str(shard),
            "--num-shards",
            str(NUM_SHARDS),
            "--out-dir",
            str(out_dir),
            "--uma-model",
            UMA_MODEL,
            "--uma-task",
            UMA_TASK,
            "--uma-fmax",
            UMA_FMAX,
            "--uma-max-steps",
            UMA_MAX_STEPS,
            "--resume",
            *extra_args,
        ]
        f = log_path.open("ab")
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=f, stderr=subprocess.STDOUT)
        f.close()
        pid_path.write_text(str(proc.pid))
        procs.append((shard, proc))
        print(f"[lbfgs] shard {shard:02d} gpu={gpu} pid={proc.pid} log={log_path}", flush=True)
        time.sleep(5)

    failed: list[tuple[int, int]] = []
    for shard, proc in procs:
        rc = proc.wait()
        print(f"[lbfgs] shard {shard:02d} exited rc={rc}", flush=True)
        if rc != 0:
            failed.append((shard, rc))
    if failed:
        raise RuntimeError(f"{stage} failed shards: {failed}")


def _run_merge(stage: str, cmd: list[str], log_name: str) -> None:
    print(f"[lbfgs] {stage}", flush=True)
    log_path = LOG_DIR / log_name
    with log_path.open("ab") as f:
        subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, check=True)
    print(f"[lbfgs] {stage} done; log={log_path}", flush=True)


def main() -> None:
    if len(GPU_LIST) < NUM_SHARDS:
        print(f"[lbfgs] GPU_LIST has {len(GPU_LIST)} entries; cycling modulo", flush=True)
    E_SYS_SHARDS.mkdir(parents=True, exist_ok=True)
    E_SLAB_SHARDS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if (REPLAY_DIR / "E_sys_lbfgs_summary.json").is_file():
        print("[lbfgs] stage 1/4+2/4 E_sys already merged; skipping", flush=True)
    else:
        if _complete_shards(E_SYS_SHARDS, "e_sys_lbfgs"):
            print("[lbfgs] stage 1/4 E_sys shards already complete; skipping shard launch", flush=True)
        else:
            _run_shards(
                "stage 1/4 E_sys L-BFGS",
                "scripts/replay/e_sys/compute_e_sys_lbfgs.py",
                E_SYS_SHARDS,
                "e_sys_lbfgs",
                [
                    "--lmdbs",
                    str(ROOT / "data" / "processed_ID" / "is2res_train.lmdb"),
                    str(ROOT / "data" / "processed_ID" / "is2res_val.lmdb"),
                ],
            )
        _run_merge(
            "stage 2/4 merge E_sys L-BFGS",
            [
                PY,
                "scripts/replay/e_sys/merge_e_sys_lbfgs_and_rebuild_gt.py",
                "--shard-dir",
                str(E_SYS_SHARDS),
                "--num-shards",
                str(NUM_SHARDS),
                "--old-gt-index",
                str(OLD_GT_INDEX),
                "--out-dir",
                str(REPLAY_DIR),
                "--uma-model",
                UMA_MODEL,
                "--uma-task",
                UMA_TASK,
                "--require-all-shards",
            ],
            "merge_e_sys_lbfgs.log",
        )
    if (REPLAY_DIR / "E_slab_only_lbfgs_summary.json").is_file():
        print("[lbfgs] stage 3/4+4/4 E_slab_only already merged; skipping", flush=True)
    else:
        if _complete_shards(E_SLAB_SHARDS, "e_slab_lbfgs"):
            print("[lbfgs] stage 3/4 E_slab_only shards already complete; skipping shard launch", flush=True)
        else:
            _run_shards(
                "stage 3/4 E_slab_only L-BFGS",
                "scripts/replay/e_sys/compute_e_slab_lbfgs.py",
                E_SLAB_SHARDS,
                "e_slab_lbfgs",
                ["--pristine-slabs", str(ROOT / "results" / "pristine_slabs" / "is2res.pkl")],
            )
        _run_merge(
            "stage 4/4 merge E_slab_only L-BFGS",
            [
                PY,
                "scripts/replay/e_sys/merge_e_slab_lbfgs.py",
                "--shard-dir",
                str(E_SLAB_SHARDS),
                "--num-shards",
                str(NUM_SHARDS),
                "--sid-index",
                str(ROOT / "results" / "pristine_slabs" / "is2res.sid_index.pkl"),
                "--out-dir",
                str(REPLAY_DIR),
                "--uma-model",
                UMA_MODEL,
                "--uma-task",
                UMA_TASK,
                "--relaxed-pristine-out",
                str(REPLAY_DIR / "pristine_slabs_lbfgs.pkl"),
                "--require-all-shards",
            ],
            "merge_e_slab_lbfgs.log",
        )
    print("[lbfgs] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[lbfgs] FAILED: {exc!r}", file=sys.stderr, flush=True)
        raise
