#!/usr/bin/env python
"""Two-stage full replay runner.

Stage 1 materializes AdsorbGen flow samples to disk as relax-job shards.
Stage 2 reads those shards and runs batched/streaming UMA L-BFGS only.
"""

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


def _dump_pickle_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _jsonable_result(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "pos_relaxed"}


@torch.no_grad()
def _build_flow_jobs(
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
            use_sde=str(getattr(args, "sample_mode", "ode")) == "sde",
            refine_final=False,
            sde_schedule=str(getattr(args, "sde_schedule", "atommof")),
            sde_alpha=float(getattr(args, "sde_alpha", 1.0)),
            sde_no_score=bool(getattr(args, "sde_no_score", False)),
            sde_mode=str(getattr(args, "sde_mode", "omatg_si")),
            si_gamma_schedule=str(getattr(args, "si_gamma_schedule", "sqrt_t1mt")),
            si_gamma_sigma=float(getattr(args, "si_gamma_sigma", 0.1)),
            si_epsilon_schedule=str(getattr(args, "si_epsilon_schedule", "vanishing_1mt")),
            si_epsilon_scale=float(getattr(args, "si_epsilon_scale", 0.01)),
            time_schedule=str(getattr(args, "time_schedule", "uniform")),
            time_schedule_beta=float(getattr(args, "time_schedule_beta", 2.0)),
            solver=str(getattr(args, "solver", "euler")),
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


def _flow_worker(worker_id: int, gpu_id: int, args_dict: dict[str, Any], task_queue, result_queue) -> None:
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
        slab_source = str(getattr(flow_model, "adsorbgen_slab_source", "initial"))
        pristine_slabs = str(getattr(flow_model, "adsorbgen_pristine_slabs", ""))
        pristine_index = str(getattr(flow_model, "adsorbgen_pristine_index", ""))
        placement_ds = [
            PlacementPriorDataset(
                p,
                prior_mode=args.prior_mode,
                max_samples=None,
                provide_ads_ref_pos=use_ads_ref,
                skip_anomaly=False,
                slab_source=slab_source,
                pristine_slabs=pristine_slabs,
                pristine_index=pristine_index,
            )
            for p in args.train_lmdb
        ]
        source_envs = [
            lmdb.open(p, subdir=False, readonly=True, lock=False, readahead=False)
            for p in args.train_lmdb
        ]
        result_queue.put(
            {
                "type": "flow_worker_ready",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "init_elapsed_sec": time.time() - t_init,
            }
        )

        while True:
            task = task_queue.get()
            if task is None:
                break
            shard_id = int(task["shard_id"])
            offset = int(task["offset"])
            n_systems = int(task["num_systems"])
            t0 = time.time()
            jobs = _build_flow_jobs(
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
            elapsed = time.time() - t0
            job_path = Path(args.jobs_dir) / f"jobs_{shard_id:06d}_offset{offset}_n{n_systems}_gpu{gpu_id}.pkl"
            _dump_pickle_atomic(job_path, {"jobs": jobs, "task": task, "settings": args_dict})
            meta = {
                "type": "flow_shard_done",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "shard_id": shard_id,
                "offset": offset,
                "num_systems": n_systems,
                "num_jobs": len(jobs),
                "elapsed_sec": elapsed,
                "jobs_per_sec": len(jobs) / elapsed if elapsed > 0 else None,
                "job_path": str(job_path),
            }
            atomic_write_json(job_path.with_suffix(".json"), meta)
            result_queue.put(meta)

        for env in source_envs:
            env.close()
        result_queue.put({"type": "flow_worker_done", "worker_id": worker_id, "gpu_id": gpu_id})
    except Exception as exc:
        result_queue.put({"type": "flow_worker_error", "worker_id": worker_id, "gpu_id": gpu_id, "error": repr(exc)})
        raise


def _relax_worker(worker_id: int, gpu_id: int, args_dict: dict[str, Any], task_queue, result_queue) -> None:
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

        t_init = time.time()
        uma = load_uma(
            args.uma_model,
            args.uma_task,
            device,
            args.uma_inference_settings,
            args.uma_internal_graph_version or None,
            args.uma_execution_mode or None,
            bool(args.uma_compile) if args.uma_compile else None,
        )
        result_queue.put(
            {
                "type": "relax_worker_ready",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "init_elapsed_sec": time.time() - t_init,
            }
        )

        while True:
            task = task_queue.get()
            if task is None:
                break
            job_path = Path(task["job_path"])
            shard_id = int(task["shard_id"])
            t0 = time.time()
            payload = _load_pickle(job_path)
            jobs = payload["jobs"]
            load_elapsed = time.time() - t0
            relax_t0 = time.time()
            results = run_optimizer(jobs, uma, args, device, "lbfgs", serial=False)
            relax_elapsed = time.time() - relax_t0
            total_elapsed = time.time() - t0
            out_json = Path(args.relax_dir) / f"relax_{shard_id:06d}_gpu{gpu_id}.json"
            summary = {
                "type": "relax_shard_done",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "shard_id": shard_id,
                "job_path": str(job_path),
                "num_jobs": len(jobs),
                "load_elapsed_sec": load_elapsed,
                "relax_elapsed_sec": relax_elapsed,
                "total_elapsed_sec": total_elapsed,
                "throughput": summarize_throughput(results, relax_elapsed),
                "end_to_end_throughput": summarize_throughput(results, total_elapsed),
                "converged": sum(1 for r in results if r["converged"]),
                "settings": args_dict,
                "results": [_jsonable_result(r) for r in results] if bool(args.write_json_rows) else [],
            }
            atomic_write_json(out_json, summary)
            if bool(args.save_result_pkl):
                out_pkl = out_json.with_suffix(".pkl")
                _dump_pickle_atomic(out_pkl, {"jobs": jobs, "results": results, "summary": summary})
            event = dict(summary)
            event.pop("results", None)
            event["out_json"] = str(out_json)
            result_queue.put(event)

        result_queue.put({"type": "relax_worker_done", "worker_id": worker_id, "gpu_id": gpu_id})
    except Exception as exc:
        result_queue.put({"type": "relax_worker_error", "worker_id": worker_id, "gpu_id": gpu_id, "error": repr(exc)})
        raise


def _run_queue(
    *,
    worker_fn,
    tasks: list[dict],
    gpus: list[int],
    args_dict: dict[str, Any],
    done_type: str,
    shard_done_type: str,
    report_path: Path,
) -> dict:
    ctx = get_context("spawn")
    task_queue = ctx.Queue(maxsize=max(len(gpus) * 2, 1))
    result_queue = ctx.Queue()
    t0 = time.time()
    procs = []
    for worker_id, gpu_id in enumerate(gpus):
        p = ctx.Process(target=worker_fn, args=(worker_id, int(gpu_id), args_dict, task_queue, result_queue))
        p.start()
        procs.append(p)

    next_task_i = 0
    sentinels_sent = 0

    def dispatch_next_or_stop() -> None:
        nonlocal next_task_i, sentinels_sent
        if next_task_i < len(tasks):
            task_queue.put(tasks[next_task_i])
            next_task_i += 1
        elif sentinels_sent < len(procs):
            task_queue.put(None)
            sentinels_sent += 1

    for _ in procs:
        dispatch_next_or_stop()

    events = []
    done_workers = 0
    completed = 0
    failed = False
    while done_workers < len(procs):
        event = result_queue.get()
        event["elapsed_since_start_sec"] = time.time() - t0
        events.append(event)
        print(json.dumps(event, sort_keys=True), flush=True)
        if event["type"] == done_type:
            done_workers += 1
        elif event["type"] == shard_done_type:
            completed += 1
            dispatch_next_or_stop()
        elif event["type"].endswith("_error"):
            failed = True
            for p in procs:
                if p.is_alive():
                    p.terminate()
            break

    for p in procs:
        p.join()

    report = {
        "settings": args_dict,
        "tasks_requested": len(tasks),
        "tasks_completed": completed,
        "workers": len(gpus),
        "wall_elapsed_sec": time.time() - t0,
        "failed": failed,
        "events": events,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def _make_flow_tasks(args) -> list[dict]:
    end = int(args.system_offset) + int(args.total_systems)
    tasks = []
    shard_id = 0
    for offset in range(int(args.system_offset), end, int(args.shard_systems)):
        n = min(int(args.shard_systems), end - offset)
        tasks.append({"shard_id": shard_id, "offset": offset, "num_systems": n})
        shard_id += 1
    return tasks


def generate(args) -> dict:
    out_dir = Path(args.out_dir)
    jobs_dir = Path(args.jobs_dir) if args.jobs_dir else out_dir / "flow_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    install_adsorbgen_imports(Path(args.repo).resolve(), Path(args.adsorbates_pkl).resolve())
    tasks = _make_flow_tasks(args)
    args_dict = vars(args).copy()
    args_dict["out_dir"] = str(out_dir)
    args_dict["jobs_dir"] = str(jobs_dir)
    atomic_write_json(out_dir / "flow_tasks.json", {"tasks": tasks, "settings": args_dict})
    return _run_queue(
        worker_fn=_flow_worker,
        tasks=tasks,
        gpus=[int(g) for g in args.gpus],
        args_dict=args_dict,
        done_type="flow_worker_done",
        shard_done_type="flow_shard_done",
        report_path=out_dir / "flow_summary.json",
    )


def _relax_tasks_from_jobs_dir(jobs_dir: Path) -> list[dict]:
    tasks = []
    for shard_id, p in enumerate(sorted(jobs_dir.glob("jobs_*.pkl"))):
        tasks.append({"shard_id": shard_id, "job_path": str(p)})
    return tasks


def relax(args) -> dict:
    out_dir = Path(args.out_dir)
    stop_marker = out_dir / "STOP_BEFORE_RELAX"
    if stop_marker.exists() and os.environ.get("ADSORBGEN_ALLOW_RELAX") != "1":
        payload = {
            "type": "relax_skipped",
            "reason": "STOP_BEFORE_RELAX marker exists",
            "marker": str(stop_marker),
            "out_dir": str(out_dir),
            "time": time.time(),
        }
        atomic_write_json(out_dir / "relax_skipped.json", payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return payload
    jobs_dir = Path(args.jobs_dir) if args.jobs_dir else out_dir / "flow_jobs"
    relax_dir = Path(args.relax_dir) if args.relax_dir else out_dir / "relax_results"
    relax_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    install_adsorbgen_imports(Path(args.repo).resolve(), Path(args.adsorbates_pkl).resolve())
    tasks = _relax_tasks_from_jobs_dir(jobs_dir)
    if not tasks:
        raise FileNotFoundError(f"no flow job shards found in {jobs_dir}")
    args_dict = vars(args).copy()
    args_dict["out_dir"] = str(out_dir)
    args_dict["jobs_dir"] = str(jobs_dir)
    args_dict["relax_dir"] = str(relax_dir)
    atomic_write_json(out_dir / "relax_tasks.json", {"tasks": tasks, "settings": args_dict})
    report = _run_queue(
        worker_fn=_relax_worker,
        tasks=tasks,
        gpus=[int(g) for g in args.gpus],
        args_dict=args_dict,
        done_type="relax_worker_done",
        shard_done_type="relax_shard_done",
        report_path=out_dir / "relax_summary.json",
    )
    _write_relax_aggregate(relax_dir, out_dir / "relax_aggregate.json")
    return report


def _write_relax_aggregate(relax_dir: Path, out_json: Path) -> None:
    docs = [json.loads(p.read_text()) for p in sorted(relax_dir.glob("relax_*.json"))]
    total_jobs = sum(int(d.get("num_jobs", 0)) for d in docs)
    total_relax = sum(float(d.get("relax_elapsed_sec", 0.0)) for d in docs)
    total_wall = sum(float(d.get("total_elapsed_sec", 0.0)) for d in docs)
    report = {
        "shards": len(docs),
        "total_jobs": total_jobs,
        "converged": sum(int(d.get("converged", 0)) for d in docs),
        "sum_relax_elapsed_sec": total_relax,
        "sum_total_elapsed_sec": total_wall,
        "sum_relax_candidates_per_sec": total_jobs / total_relax if total_relax > 0 else None,
        "sum_total_candidates_per_sec": total_jobs / total_wall if total_wall > 0 else None,
    }
    atomic_write_json(out_json, report)


def launch(args) -> None:
    generate(args)
    relax(args)


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--repo", default="/home/irteam/AdsorbGen")
    ap.add_argument("--adsorbates-pkl", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-lmdb", nargs="+", required=True)
    ap.add_argument("--selected-systems", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs-dir", default="")
    ap.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--seed", type=int, default=20260526)


def _add_flow(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--total-systems", type=int, default=2048)
    ap.add_argument("--system-offset", type=int, default=0)
    ap.add_argument("--shard-systems", type=int, default=256)
    ap.add_argument("--num-placements", type=int, default=10)
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--flow-batch-size", type=int, default=64)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--langevin-uma-model", default="uma-s-1p2")
    ap.add_argument("--langevin-uma-task", default="oc20")
    ap.add_argument("--sample-mode", choices=["ode", "sde"], default="ode")
    ap.add_argument("--solver", choices=["euler", "heun"], default="euler")
    ap.add_argument("--sde-mode", choices=["atommof", "omatg_si"], default="omatg_si")
    ap.add_argument("--sde-schedule", choices=["atommof", "zero_ends"], default="atommof")
    ap.add_argument("--sde-alpha", type=float, default=1.0)
    ap.add_argument("--sde-no-score", action="store_true")
    ap.add_argument("--si-gamma-schedule", default="sqrt_t1mt")
    ap.add_argument("--si-gamma-sigma", type=float, default=0.1)
    ap.add_argument("--si-epsilon-schedule", default="vanishing_1mt")
    ap.add_argument("--si-epsilon-scale", type=float, default=0.01)
    ap.add_argument("--time-schedule", choices=["uniform", "high_t_power", "low_t_power", "beta_train"], default="uniform")
    ap.add_argument("--time-schedule-beta", type=float, default=2.0)


def _add_relax(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--relax-dir", default="")
    ap.add_argument("--uma-model", default="uma-s-1p2")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--uma-inference-settings", default="default", choices=["default", "turbo", "traineval"])
    ap.add_argument("--uma-internal-graph-version", type=int, default=0)
    ap.add_argument("--uma-execution-mode", default="")
    ap.add_argument("--uma-compile", action="store_true")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-atoms", type=int, default=32768)
    ap.add_argument("--maxstep", type=float, default=0.04)
    ap.add_argument("--lbfgs-memory", type=int, default=50)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    ap.add_argument("--lbfgs-history-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-position-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-curvature-guard", choices=["abs", "positive", "ase"], default="abs")
    ap.add_argument("--lbfgs-gpu-history-guard", action="store_true")
    ap.add_argument("--lbfgs-keep-survivors-on-gpu", action="store_true")
    ap.add_argument("--lbfgs-streaming", action="store_true")
    ap.add_argument("--lbfgs-check-interval", type=int, default=10)
    ap.add_argument("--lbfgs-stream-sort", action="store_true")
    ap.add_argument("--fire-dt", type=float, default=0.1)
    ap.add_argument("--fire-dt-max", type=float, default=1.0)
    ap.add_argument("--cg-step-size", type=float, default=0.04)
    ap.add_argument("--save-result-pkl", action="store_true")
    ap.add_argument("--write-json-rows", action="store_true")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="materialize flow samples as relax-job shards")
    _add_common(gen)
    _add_flow(gen)

    rel = sub.add_parser("relax", help="run UMA L-BFGS from materialized flow shards")
    _add_common(rel)
    _add_relax(rel)

    both = sub.add_parser("launch", help="run generate and then relax in one foreground process")
    _add_common(both)
    _add_flow(both)
    _add_relax(both)
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.cmd == "generate":
        generate(args)
    elif args.cmd == "relax":
        relax(args)
    elif args.cmd == "launch":
        launch(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
