#!/usr/bin/env python
"""Run one relaxation-only throughput benchmark on stored two-stage jobs."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geoopt import atomic_write_json, load_uma, run_optimizer  # noqa: E402


class CountingUMA:
    def __init__(self, inner):
        self.inner = inner
        self.batch_calls = 0
        self.graph_evals = 0
        self.atom_evals = 0
        self.max_graphs_per_call = 0
        self.max_atoms_per_call = 0

    def __call__(self, batch):
        n_atoms = int(getattr(batch, "positions").shape[0])
        energy = getattr(batch, "energy", None)
        if energy is not None:
            n_graphs = int(energy.reshape(-1).numel())
        else:
            n_graphs = int(getattr(batch, "batch_ptr").reshape(-1).numel()) - 1
        self.batch_calls += 1
        self.graph_evals += n_graphs
        self.atom_evals += n_atoms
        self.max_graphs_per_call = max(self.max_graphs_per_call, n_graphs)
        self.max_atoms_per_call = max(self.max_atoms_per_call, n_atoms)
        return self.inner(batch)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def load_jobs(flow_jobs_dir: Path, candidate_limit: int) -> list[dict]:
    jobs: list[dict] = []
    for path in sorted(flow_jobs_dir.glob("jobs_*.pkl")):
        with path.open("rb") as f:
            shard = pickle.load(f)
        if isinstance(shard, dict):
            shard = shard["jobs"]
        jobs.extend(shard)
        if len(jobs) >= candidate_limit:
            return jobs[:candidate_limit]
    return jobs


def poll_gpu(gpu_index: int, stop: threading.Event, samples: list[dict]) -> None:
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_index),
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
            if out:
                mem, util = [int(x.strip()) for x in out.split(",")[:2]]
                samples.append({"t": time.time(), "mem_mb": mem, "util_pct": util})
        except Exception:
            pass
        stop.wait(0.5)


def pct(values: list[int], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow-jobs-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--gpu-index", type=int, required=True)
    ap.add_argument("--candidate-limit", type=int, required=True)
    ap.add_argument("--max-atoms", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--uma-model", default="uma-s-1p2")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--uma-inference-settings", default="default", choices=["default", "turbo", "traineval"])
    ap.add_argument("--uma-internal-graph-version", type=int, default=0)
    ap.add_argument("--uma-execution-mode", default="")
    ap.add_argument("--uma-compile", action="store_true")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--maxstep", type=float, default=0.04)
    ap.add_argument("--lbfgs-memory", type=int, default=50)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-check-interval", type=int, default=10)
    ap.add_argument("--lbfgs-streaming", dest="lbfgs_streaming", action="store_true", default=True)
    ap.add_argument("--no-lbfgs-streaming", dest="lbfgs_streaming", action="store_false")
    ap.add_argument("--lbfgs-gpu-history-guard", action="store_true")
    ap.add_argument("--lbfgs-keep-survivors-on-gpu", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out_json)
    payload = {
        "gpu_index": int(args.gpu_index),
        "candidate_limit": int(args.candidate_limit),
        "max_atoms": int(args.max_atoms),
        "max_steps": int(args.max_steps),
        "status": "started",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(out_path, payload)

    stop = threading.Event()
    samples: list[dict] = []
    poller = threading.Thread(target=poll_gpu, args=(args.gpu_index, stop, samples), daemon=True)
    poller.start()
    t0 = time.time()

    try:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        jobs = load_jobs(Path(args.flow_jobs_dir), int(args.candidate_limit))
        atom_counts = [int(len(j["relax_input"]["numbers"])) for j in jobs]
        opt_args = SimpleNamespace(
            fmax=float(args.fmax),
            max_steps=int(args.max_steps),
            max_atoms=int(args.max_atoms),
            maxstep=float(args.maxstep),
            lbfgs_memory=int(args.lbfgs_memory),
            lbfgs_alpha=float(args.lbfgs_alpha),
            lbfgs_damping=float(args.lbfgs_damping),
            lbfgs_check_interval=int(args.lbfgs_check_interval),
            lbfgs_streaming=bool(args.lbfgs_streaming),
            lbfgs_gpu_history_guard=bool(args.lbfgs_gpu_history_guard),
            lbfgs_keep_survivors_on_gpu=bool(args.lbfgs_keep_survivors_on_gpu),
        )
        t_load0 = time.time()
        uma = CountingUMA(load_uma(
            str(args.uma_model),
            str(args.uma_task),
            device,
            str(args.uma_inference_settings),
            int(args.uma_internal_graph_version) or None,
            str(args.uma_execution_mode) or None,
            bool(args.uma_compile) if bool(args.uma_compile) else None,
        ))
        load_elapsed = time.time() - t_load0
        torch.cuda.reset_peak_memory_stats(device)
        t_relax0 = time.time()
        results = run_optimizer(jobs, uma, opt_args, device, "lbfgs", serial=False)
        relax_elapsed = time.time() - t_relax0
        result_errors = [r.get("error") for r in results if r.get("error")]
        elapsed = time.time() - t0
        stop.set()
        poller.join(timeout=2)
        mems = [s["mem_mb"] for s in samples]
        utils = [s["util_pct"] for s in samples]
        payload.update(
            {
                "status": "ok",
                "elapsed_sec": float(elapsed),
                "uma_load_elapsed_sec": float(load_elapsed),
                "relax_elapsed_sec": float(relax_elapsed),
                "completed_results": len(results),
                "result_error_count": int(len(result_errors)),
                "first_result_error": str(result_errors[0]) if result_errors else None,
                "input_atoms": int(sum(atom_counts)),
                "mean_atoms": float(np.mean(atom_counts)) if atom_counts else None,
                "max_candidate_atoms": int(max(atom_counts)) if atom_counts else None,
                "lbfgs_streaming": bool(args.lbfgs_streaming),
                "lbfgs_gpu_history_guard": bool(args.lbfgs_gpu_history_guard),
                "lbfgs_keep_survivors_on_gpu": bool(args.lbfgs_keep_survivors_on_gpu),
                "throughput_candidates_per_sec": len(results) / relax_elapsed if relax_elapsed > 0 else None,
                "mlip_batch_calls": int(uma.batch_calls),
                "mlip_batch_calls_per_sec": uma.batch_calls / relax_elapsed if relax_elapsed > 0 else None,
                "mlip_graph_evals": int(uma.graph_evals),
                "mlip_graph_evals_per_sec": uma.graph_evals / relax_elapsed if relax_elapsed > 0 else None,
                "mlip_atom_evals": int(uma.atom_evals),
                "mlip_atom_evals_per_sec": uma.atom_evals / relax_elapsed if relax_elapsed > 0 else None,
                "max_graphs_per_mlip_call": int(uma.max_graphs_per_call),
                "max_atoms_per_mlip_call": int(uma.max_atoms_per_call),
                "converged": int(sum(1 for r in results if r.get("converged"))),
                "converged_rate": float(sum(1 for r in results if r.get("converged")) / max(len(results), 1)),
                "peak_torch_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
                "peak_nvidia_mem_mb": int(max(mems)) if mems else None,
                "avg_gpu_util_pct": float(np.mean(utils)) if utils else None,
                "p95_gpu_util_pct": pct(utils, 95),
                "samples": len(samples),
            }
        )
    except BaseException as exc:
        stop.set()
        poller.join(timeout=2)
        mems = [s["mem_mb"] for s in samples]
        utils = [s["util_pct"] for s in samples]
        payload.update(
            {
                "status": "error",
                "elapsed_sec": float(time.time() - t0),
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=20),
                "peak_nvidia_mem_mb": int(max(mems)) if mems else None,
                "avg_gpu_util_pct": float(np.mean(utils)) if utils else None,
                "p95_gpu_util_pct": pct(utils, 95),
                "samples": len(samples),
            }
        )
    finally:
        atomic_write_json(out_path, payload)


if __name__ == "__main__":
    main()
