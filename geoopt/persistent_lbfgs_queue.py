#!/usr/bin/env python
"""Persistent one-process-per-GPU batched LBFGS queue runner."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lmdb
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geoopt import (  # noqa: E402
    atomic_write_json,
    install_adsorbgen_imports,
    load_model_from_ckpt,
    load_selected_representatives,
    load_uma,
    read_entry,
    run_optimizer,
    summarize_throughput,
)


def frozen_key(x):
    if isinstance(x, (list, tuple)):
        return tuple(frozen_key(v) for v in x)
    return x


@torch.no_grad()
def build_relax_jobs_from_worker_context(
    *,
    args: SimpleNamespace,
    selected_all: list[dict],
    system_offset: int,
    num_systems: int,
    device: torch.device,
    flow_model,
    flow_cfg,
    placement_ds: list,
    source_envs: list[lmdb.Environment],
    use_ads_ref: bool,
    langevin_force_model=None,
) -> list[dict]:
    from adsorbgen.data.dataset import collate_displacement
    from adsorbgen.flow import euler_sample
    from adsorbgen.replay.eval import _runtime_movable_mask

    selected = selected_all[int(system_offset): int(system_offset) + int(num_systems)]
    tasks = []
    for local_sys_i, rep in enumerate(selected):
        absolute_sys_i = int(system_offset) + local_sys_i
        for sample_i in range(int(args.num_placements)):
            global_i = absolute_sys_i * int(args.num_placements) + sample_i
            tasks.append((global_i, absolute_sys_i, sample_i, rep))

    jobs: list[dict] = []
    for start in range(0, len(tasks), int(args.flow_batch_size)):
        chunk = tasks[start:start + int(args.flow_batch_size)]
        samples = []
        metas = []
        for global_i, sys_i, sample_i, rep in chunk:
            seed_i = (int(args.seed) + int(global_i)) & 0xFFFF_FFFF
            np.random.seed(seed_i)
            random.seed(seed_i)
            sample = placement_ds[int(rep["lmdb_id"])][int(rep["raw_idx"])]
            samples.append(sample)
            metas.append((global_i, sys_i, sample_i, rep, sample))

        batch = collate_displacement(samples)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        movable = _runtime_movable_mask(flow_model, batch)

        def fwd(x_t, t, _batch=batch, _movable=movable):
            extra = {}
            if use_ads_ref:
                extra["ads_ref_pos"] = _batch["ads_ref_pos"]
            if langevin_force_model is not None:
                extra["mlip_force"] = langevin_force_model(
                    x_t.detach(),
                    _batch["cell"],
                    _batch["atomic_numbers"],
                    _batch["pad_mask"],
                )
                extra["langevin_prediction_type"] = flow_cfg.prediction_type
            return flow_model(
                pos=_batch["pos"],
                x_t=x_t,
                t=t,
                atomic_numbers=_batch["atomic_numbers"],
                tags=_batch["tags"],
                movable_mask=_movable,
                pad_mask=_batch["pad_mask"],
                cell=_batch["cell"],
                **extra,
            )

        x_out = euler_sample(
            fwd,
            batch["pos"],
            movable,
            batch["pad_mask"],
            flow_cfg,
            num_steps=int(args.flow_steps),
            use_sde=False,
            refine_final=False,
        )
        for i, (global_i, sys_i, sample_i, rep, sample) in enumerate(metas):
            n = int(batch["pad_mask"][i].sum().item())
            cell = batch["cell"][i].detach().cpu().numpy()
            if cell.ndim == 3:
                cell = cell[0]
            source_entry = read_entry(source_envs[int(rep["lmdb_id"])], int(rep["raw_idx"]))
            ads_id = int(batch["ads_id"][i].item()) if "ads_id" in batch else int(sample["ads_id"].item())
            jobs.append(
                {
                    "global_i": int(global_i),
                    "system_i": int(sys_i),
                    "sample_i": int(sample_i),
                    "system_key": list(rep["system_key"]),
                    "sid": int(rep["sid"]),
                    "E_sys_ref": float(rep["E_sys_ref"]),
                    "source_energy": float(source_entry.get("y", np.nan))
                    if isinstance(source_entry, dict) else float("nan"),
                    "ads_id": int(ads_id),
                    "pos_ref": batch["pos"][i, :n].detach().cpu().numpy().astype(np.float64),
                    "pos_gt": batch["pos_relaxed"][i, :n].detach().cpu().numpy().astype(np.float64),
                    "relax_input": {
                        "numbers": batch["atomic_numbers"][i, :n].detach().cpu().numpy().astype(np.int64),
                        "tags": batch["tags"][i, :n].detach().cpu().numpy().astype(np.int64),
                        "fixed": batch["fixed"][i, :n].detach().cpu().numpy().astype(np.int64),
                        "cell": cell.astype(np.float32),
                        "pos_pred": x_out[i, :n].detach().cpu().numpy().astype(np.float64),
                    },
                }
            )
    return jobs


def jsonable_result(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "pos_relaxed"}


def worker_main(worker_id: int, gpu_id: int, args_dict: dict[str, Any], task_queue, result_queue) -> None:
    args = SimpleNamespace(**args_dict)
    try:
        install_adsorbgen_imports(Path(args.repo).resolve(), Path(args.adsorbates_pkl).resolve())
        os.environ["ADSGEN_ROOT"] = str(Path(args.repo).resolve())
        os.environ["ADSORBATES_PKL"] = str(Path(args.adsorbates_pkl).resolve())
        random.seed(int(args.seed) + worker_id)
        np.random.seed(int(args.seed) + worker_id)
        torch.manual_seed(int(args.seed) + worker_id)
        torch.cuda.set_device(int(gpu_id))
        device = torch.device(f"cuda:{int(gpu_id)}")

        from adsorbgen.data.dataset import PlacementPriorDataset
        from adsorbgen.evaluation.energy import UMAForce
        from adsorbgen.replay.eval import _model_cfg

        t_init = time.time()
        selected_all = load_selected_representatives(Path(args.selected_systems))
        flow_model, flow_cfg = load_model_from_ckpt(Path(args.ckpt), device)
        flow_model_cfg = _model_cfg(flow_model)
        use_ads_ref = bool(getattr(flow_model_cfg, "use_ads_ref_pos", False))
        langevin_force_model = None
        if bool(getattr(flow_model_cfg, "use_langevin_param", False)):
            if str(getattr(flow_model_cfg, "langevin_eval_on", "x_t")) != "x_t":
                raise ValueError("Only langevin_eval_on='x_t' is implemented")
            langevin_force_model = UMAForce(
                model_name=args.langevin_uma_model,
                task_name=args.langevin_uma_task,
                device=str(device),
            )
        placement_ds = [
            PlacementPriorDataset(
                p,
                prior_mode=args.prior_mode,
                max_samples=None,
                provide_ads_ref_pos=use_ads_ref,
                skip_anomaly=False,
            )
            for p in args.train_lmdb
        ]
        source_envs = [
            lmdb.open(p, subdir=False, readonly=True, lock=False, readahead=False)
            for p in args.train_lmdb
        ]
        uma = load_uma(args.uma_model, args.uma_task, device)
        init_elapsed = time.time() - t_init
        result_queue.put({"type": "worker_ready", "worker_id": worker_id, "gpu_id": gpu_id, "init_elapsed_sec": init_elapsed})

        while True:
            task = task_queue.get()
            if task is None:
                break
            offset = int(task["offset"])
            n_systems = int(task["num_systems"])
            chunk_id = int(task["chunk_id"])
            t0 = time.time()
            flow_t0 = time.time()
            jobs = build_relax_jobs_from_worker_context(
                args=args,
                selected_all=selected_all,
                system_offset=offset,
                num_systems=n_systems,
                device=device,
                flow_model=flow_model,
                flow_cfg=flow_cfg,
                placement_ds=placement_ds,
                source_envs=source_envs,
                use_ads_ref=use_ads_ref,
                langevin_force_model=langevin_force_model,
            )
            flow_elapsed = time.time() - flow_t0
            relax_t0 = time.time()
            results = run_optimizer(jobs, uma, args, device, "lbfgs", serial=False)
            relax_elapsed = time.time() - relax_t0
            total_elapsed = time.time() - t0

            payload = {
                "worker_id": int(worker_id),
                "gpu_id": int(gpu_id),
                "chunk_id": chunk_id,
                "offset": offset,
                "num_systems": n_systems,
                "num_jobs": len(jobs),
                "flow_elapsed_sec": flow_elapsed,
                "relax_elapsed_sec": relax_elapsed,
                "total_elapsed_sec": total_elapsed,
                "throughput": summarize_throughput(results, relax_elapsed),
                "end_to_end_throughput": summarize_throughput(results, total_elapsed),
                "converged": sum(1 for r in results if r["converged"]),
                "settings": args_dict,
                "results": [jsonable_result(r) for r in results] if bool(args.write_json_rows) else [],
            }
            out_json = Path(args.out_dir) / f"chunk_{chunk_id:06d}_offset{offset}_n{n_systems}_gpu{gpu_id}.json"
            atomic_write_json(out_json, payload)
            if bool(args.save_result_pkl):
                out_pkl = Path(args.out_dir) / f"chunk_{chunk_id:06d}_offset{offset}_n{n_systems}_gpu{gpu_id}.pkl"
                with out_pkl.open("wb") as f:
                    pickle.dump({"jobs": jobs, "results": results, "summary": payload}, f, protocol=pickle.HIGHEST_PROTOCOL)
            result_queue.put(
                {
                    "type": "chunk_done",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "chunk_id": chunk_id,
                    "offset": offset,
                    "num_jobs": len(jobs),
                    "flow_elapsed_sec": flow_elapsed,
                    "relax_elapsed_sec": relax_elapsed,
                    "total_elapsed_sec": total_elapsed,
                    "out_json": str(out_json),
                }
            )

        for env in source_envs:
            env.close()
        result_queue.put({"type": "worker_done", "worker_id": worker_id, "gpu_id": gpu_id})
    except Exception as exc:
        result_queue.put({"type": "worker_error", "worker_id": worker_id, "gpu_id": gpu_id, "error": repr(exc)})
        raise


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/irteam/AdsorbGen")
    ap.add_argument("--adsorbates-pkl", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-lmdb", nargs="+", required=True)
    ap.add_argument("--selected-systems", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--seed", type=int, default=20260526)
    ap.add_argument("--total-systems", type=int, default=2048)
    ap.add_argument("--system-offset", type=int, default=0)
    ap.add_argument("--chunk-systems", type=int, default=256)
    ap.add_argument("--num-placements", type=int, default=1)
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--flow-batch-size", type=int, default=64)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--uma-model", default="uma-s-1p2")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--langevin-uma-model", default="uma-s-1p2")
    ap.add_argument("--langevin-uma-task", default="oc20")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-atoms", type=int, default=32768)
    ap.add_argument("--maxstep", type=float, default=0.04)
    ap.add_argument("--lbfgs-memory", type=int, default=50)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    ap.add_argument("--lbfgs-history-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-position-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-curvature-guard", choices=["abs", "positive", "ase"], default="abs")
    ap.add_argument("--lbfgs-streaming", action="store_true")
    ap.add_argument("--lbfgs-check-interval", type=int, default=10)
    ap.add_argument("--lbfgs-stream-sort", action="store_true")
    ap.add_argument("--fire-dt", type=float, default=0.1)
    ap.add_argument("--fire-dt-max", type=float, default=1.0)
    ap.add_argument("--cg-step-size", type=float, default=0.04)
    ap.add_argument("--save-result-pkl", action="store_true")
    ap.add_argument("--write-json-rows", action="store_true")
    return ap


def main() -> None:
    args = parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    install_adsorbgen_imports(Path(args.repo).resolve(), Path(args.adsorbates_pkl).resolve())

    chunks = []
    chunk_id = 0
    end = int(args.system_offset) + int(args.total_systems)
    for offset in range(int(args.system_offset), end, int(args.chunk_systems)):
        n = min(int(args.chunk_systems), end - offset)
        chunks.append({"chunk_id": chunk_id, "offset": offset, "num_systems": n})
        chunk_id += 1

    ctx = get_context("spawn")
    task_queue = ctx.Queue(maxsize=max(len(args.gpus) * 2, 1))
    result_queue = ctx.Queue()
    args_dict = vars(args).copy()
    args_dict["out_dir"] = str(out_dir)
    t0 = time.time()
    procs = []
    for worker_id, gpu_id in enumerate(args.gpus):
        p = ctx.Process(target=worker_main, args=(worker_id, int(gpu_id), args_dict, task_queue, result_queue))
        p.start()
        procs.append(p)

    next_chunk_i = 0
    sentinels_sent = 0

    def dispatch_next_or_stop() -> None:
        nonlocal next_chunk_i, sentinels_sent
        if next_chunk_i < len(chunks):
            task_queue.put(chunks[next_chunk_i])
            next_chunk_i += 1
        elif sentinels_sent < len(procs):
            task_queue.put(None)
            sentinels_sent += 1

    for _ in procs:
        dispatch_next_or_stop()

    events = []
    done_workers = 0
    completed_chunks = 0
    while done_workers < len(procs):
        event = result_queue.get()
        event["elapsed_since_start_sec"] = time.time() - t0
        events.append(event)
        print(json.dumps(event, sort_keys=True), flush=True)
        if event["type"] == "worker_done":
            done_workers += 1
        elif event["type"] == "chunk_done":
            completed_chunks += 1
            dispatch_next_or_stop()
        elif event["type"] == "worker_error":
            break

    for p in procs:
        p.join()

    chunk_files = sorted(out_dir.glob("chunk_*.json"))
    chunk_docs = [json.loads(p.read_text()) for p in chunk_files]
    total_jobs = sum(int(d["num_jobs"]) for d in chunk_docs)
    total_relax_elapsed = sum(float(d["relax_elapsed_sec"]) for d in chunk_docs)
    wall_elapsed = time.time() - t0
    report = {
        "settings": args_dict,
        "chunks_requested": len(chunks),
        "chunks_completed": completed_chunks,
        "workers": len(args.gpus),
        "events": events,
        "total_jobs": total_jobs,
        "wall_elapsed_sec": wall_elapsed,
        "effective_candidates_per_sec": total_jobs / wall_elapsed if wall_elapsed > 0 else None,
        "sum_chunk_relax_elapsed_sec": total_relax_elapsed,
        "sum_chunk_relax_candidates_per_sec": total_jobs / total_relax_elapsed if total_relax_elapsed > 0 else None,
        "converged": sum(int(d.get("converged", 0)) for d in chunk_docs),
    }
    atomic_write_json(out_dir / "persistent_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
