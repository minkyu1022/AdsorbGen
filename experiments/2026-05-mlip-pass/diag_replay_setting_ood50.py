#!/usr/bin/env python
"""OOD50 replay-setting diagnostics.

For each candidate:
  generator output -> UMA single-point energy gap -> batched LBFGS relaxation.

This intentionally uses ``geoopt.run_optimizer`` so the relaxation path matches
the replay batched LBFGS path rather than the ASE-LBFGS pass@k path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lmdb
import numpy as np
import torch
from tqdm.auto import tqdm

REPO = Path(os.environ.get("ADSGEN_ROOT", "/home1/irteam/AdsorbGen")).resolve()
GEOOPT = REPO / "geoopt"
MLIP_PASS = REPO / "experiments" / "2026-05-mlip-pass"
for p in (REPO, GEOOPT, MLIP_PASS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement  # noqa: E402
from adsorbgen.evaluation.energy import UMAEnergy, UMAForce  # noqa: E402
from adsorbgen.evaluation.metrics import load_pristine_context, _score_record_anomaly  # noqa: E402
from adsorbgen.flow import FKSteeringConfig, euler_sample  # noqa: E402
from adsorbgen.replay.eval import _model_cfg, _runtime_movable_mask  # noqa: E402
from eval_mlip_pass_lbfgs_ood50 import load_reference_by_system, map_system_to_raw_idx  # noqa: E402
from geoopt import (  # noqa: E402
    copy_uma_outputs,
    load_model_from_ckpt,
    load_uma,
    make_nvbatch,
    run_optimizer,
)


def _read_lmdb_entry(env: lmdb.Environment, idx: int):
    with env.begin() as txn:
        raw = txn.get(str(int(idx)).encode("ascii"))
    if raw is None:
        raise KeyError(f"missing LMDB key {idx}")
    return pickle.loads(raw)


def _to_numpy(x, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def _cell3(cell) -> np.ndarray:
    arr = _to_numpy(cell, np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def _data_field(data, name: str, default=None):
    if hasattr(data, name):
        return getattr(data, name)
    if isinstance(data, dict):
        return data.get(name, default)
    return default


def load_selected_systems(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    return [str(x) for x in payload["systems"]]


def load_generated_positions(results_dir: Path) -> dict[int, np.ndarray]:
    paths = sorted(results_dir.glob("**/relaxed_positions.npz"))
    if not paths:
        paths = sorted(results_dir.glob("**/relaxed_pos_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no AdsorbDiff relaxed position npz found under {results_dir}")
    out: dict[int, np.ndarray] = {}
    for path in paths:
        data = np.load(path, allow_pickle=True)
        for sid, xyz in zip(data["ids"], data["pos"]):
            out[int(str(sid))] = np.asarray(xyz, dtype=np.float64)
    return out


def select_tasks(args) -> tuple[list[str], dict[str, int], list[tuple[int, str, int, int]]]:
    systems = load_selected_systems(Path(args.selected_systems))
    raw_idx = map_system_to_raw_idx(args.lmdb, set(systems))
    tasks = []
    for sys_i, sk in enumerate(systems):
        if bool(getattr(args, "shard_by_system", False)) and sys_i % int(args.num_shards) != int(args.shard_idx):
            continue
        for sample_i in range(int(args.num_samples)):
            global_i = sys_i * int(args.num_samples) + sample_i
            if bool(getattr(args, "shard_by_system", False)) or global_i % int(args.num_shards) == int(args.shard_idx):
                tasks.append((global_i, sk, sample_i, int(raw_idx[sk])))
    return systems, raw_idx, tasks


@torch.no_grad()
def build_model_or_random_jobs(args, device: torch.device, refs: dict[str, dict]) -> list[dict]:
    systems, _, tasks = select_tasks(args)
    flow_model = None
    flow_cfg = None
    use_ads_ref = False
    langevin_force_model = None
    fk_energy_model = None
    slab_source = str(args.slab_source)
    pristine_slabs = str(args.placement_pristine_slabs)
    pristine_index = str(args.placement_pristine_index)

    if args.mode == "model":
        flow_model, flow_cfg = load_model_from_ckpt(Path(args.ckpt), device)
        model_cfg = _model_cfg(flow_model)
        use_ads_ref = bool(getattr(model_cfg, "use_ads_ref_pos", False))
        if args.slab_source == "auto":
            slab_source = str(getattr(flow_model, "adsorbgen_slab_source", "initial"))
            pristine_slabs = str(getattr(flow_model, "adsorbgen_pristine_slabs", ""))
            pristine_index = str(getattr(flow_model, "adsorbgen_pristine_index", ""))
        if bool(getattr(model_cfg, "use_langevin_param", False)):
            if str(getattr(model_cfg, "langevin_eval_on", "x_t")) != "x_t":
                raise ValueError("Only langevin_eval_on='x_t' is implemented")
            langevin_force_model = UMAForce(
                model_name=args.langevin_uma_model,
                task_name=args.langevin_uma_task,
                device=str(device),
            )
        if args.sample_mode == "sde_fk":
            fk_energy_model = UMAEnergy(
                model_name=args.fk_uma_model,
                task_name=args.fk_uma_task,
                device=str(device),
                normalize_per_atom=bool(args.fk_normalize_per_atom),
            )
    elif args.mode == "random":
        if args.slab_source == "auto":
            slab_source = "initial"
    else:
        raise ValueError(args.mode)

    ds = PlacementPriorDataset(
        args.lmdb,
        prior_mode=args.prior_mode,
        max_samples=None,
        provide_ads_ref_pos=use_ads_ref,
        skip_anomaly=False,
        slab_source=slab_source,
        pristine_slabs=pristine_slabs,
        pristine_index=pristine_index,
    )

    jobs: list[dict] = []
    for start in tqdm(
        range(0, len(tasks), int(args.flow_batch_size)),
        desc=f"[{args.label} shard{args.shard_idx}] generate",
        unit="batch",
        dynamic_ncols=True,
    ):
        chunk = tasks[start:start + int(args.flow_batch_size)]
        samples = []
        metas = []
        for global_i, sk, sample_i, raw_idx in chunk:
            seed_i = (int(args.seed) + int(global_i)) & 0xFFFF_FFFF
            np.random.seed(seed_i)
            random.seed(seed_i)
            samples.append(ds[int(raw_idx)])
            metas.append((global_i, sk, sample_i, raw_idx))

        batch = collate_displacement(samples)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if args.mode == "random":
            x_out = batch["pos"]
        else:
            assert flow_model is not None and flow_cfg is not None
            movable = _runtime_movable_mask(flow_model, batch)
            want_si_eta = bool(args.sample_mode in ("sde", "sde_fk") and args.sde_mode == "omatg_si")

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
                if want_si_eta:
                    extra["return_si_eta"] = True
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

            fk_cfg = None
            if args.sample_mode == "sde_fk":
                assert fk_energy_model is not None
                if int(args.flow_batch_size) % int(args.fk_num_particles) != 0:
                    raise ValueError("--flow-batch-size must be divisible by --fk-num-particles")
                if int(args.num_samples) % int(args.fk_num_particles) != 0:
                    raise ValueError("--num-samples must be divisible by --fk-num-particles for grouped FK")

                def fk_energy_fn(x_pred, pad_mask, movable_mask, _batch=batch):
                    return fk_energy_model(
                        x_pred,
                        _batch["cell"],
                        _batch["atomic_numbers"],
                        pad_mask,
                    )

                fk_cfg = FKSteeringConfig(
                    num_particles=int(args.fk_num_particles),
                    energy_fn=fk_energy_fn,
                    fk_lambda=float(args.fk_lambda),
                    resampling_interval=int(args.fk_resample_interval),
                    fk_start_time=float(args.fk_start_time),
                    potential_mode=str(args.fk_potential_mode),
                )

            x_out = euler_sample(
                fwd,
                batch["pos"],
                movable,
                batch["pad_mask"],
                flow_cfg,
                num_steps=int(args.flow_steps),
                use_sde=args.sample_mode in ("sde", "sde_fk"),
                refine_final=False,
                fk_steering=fk_cfg,
                sde_schedule=args.sde_schedule,
                sde_alpha=float(args.sde_alpha),
                sde_no_score=bool(args.sde_no_score),
                sde_mode=args.sde_mode,
                si_gamma_schedule=args.si_gamma_schedule,
                si_gamma_sigma=float(args.si_gamma_sigma),
                si_epsilon_schedule=args.si_epsilon_schedule,
                si_epsilon_scale=float(args.si_epsilon_scale),
                time_schedule=args.time_schedule,
                time_schedule_beta=float(args.time_schedule_beta),
                solver=args.solver,
            )

        for i, (global_i, sk, sample_i, raw_idx) in enumerate(metas):
            n = int(batch["pad_mask"][i].sum().item())
            cell = batch["cell"][i].detach().cpu().numpy()
            if cell.ndim == 3:
                cell = cell[0]
            ads_id = int(batch["ads_id"][i].item()) if "ads_id" in batch else int(samples[i]["ads_id"].item())
            jobs.append(
                {
                    "global_i": int(global_i),
                    "system_i": systems.index(sk),
                    "sample_i": int(sample_i),
                    "system_key": sk,
                    "sid": -1,
                    "E_sys_ref": float(refs[sk]["E_sys_ref"]),
                    "E_ads_ref": float(refs[sk]["E_ads_ref"]),
                    "ads_id": int(ads_id),
                    "raw_idx": int(raw_idx),
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


def build_adsorbdiff_jobs(args, refs: dict[str, dict]) -> list[dict]:
    meta = json.loads(Path(args.adsorbdiff_metadata).read_text())
    generated = load_generated_positions(Path(args.adsorbdiff_results_dir))
    rows_meta = {int(r["global_i"]): r for r in meta["rows"]}
    cand_env = lmdb.open(str(args.adsorbdiff_lmdb or meta["lmdb"]), subdir=False, readonly=True, lock=False, readahead=False)
    src_env = lmdb.open(str(meta["source_lmdb"]), subdir=False, readonly=True, lock=False, readahead=False)
    jobs: list[dict] = []
    task_ids = sorted(i for i in generated if i % int(args.num_shards) == int(args.shard_idx))
    for global_i in tqdm(task_ids, desc=f"[{args.label} shard{args.shard_idx}] load", unit="cand", dynamic_ncols=True):
        m = rows_meta[int(global_i)]
        sk = str(m["system_key"])
        data = _read_lmdb_entry(cand_env, int(m["lmdb_idx"]))
        src = _read_lmdb_entry(src_env, int(m["raw_idx"]))
        tags = _to_numpy(_data_field(data, "tags"), np.int64)
        numbers = _to_numpy(_data_field(data, "atomic_numbers"), np.int64)
        fixed = _to_numpy(_data_field(data, "fixed", np.zeros_like(tags)), np.int64)
        cell = _cell3(_data_field(data, "cell"))
        pos_ref = _to_numpy(_data_field(data, "pos"), np.float64)
        pos_gt = _to_numpy(src["pos_relaxed"], np.float64)
        pos_pred = np.asarray(generated[int(global_i)], dtype=np.float64)
        if pos_pred.shape != pos_ref.shape:
            raise ValueError(f"global_i={global_i} generated={pos_pred.shape} input={pos_ref.shape}")
        jobs.append(
            {
                "global_i": int(global_i),
                "system_i": int(global_i) // int(args.num_samples),
                "sample_i": int(m["sample_i"]),
                "system_key": sk,
                "sid": int(global_i),
                "E_sys_ref": float(refs[sk]["E_sys_ref"]),
                "E_ads_ref": float(refs[sk]["E_ads_ref"]),
                "ads_id": int(m.get("ads_id", _data_field(data, "ads_id", -1))),
                "raw_idx": int(m["raw_idx"]),
                "pos_ref": pos_ref,
                "pos_gt": pos_gt,
                "relax_input": {
                    "numbers": numbers,
                    "tags": tags,
                    "fixed": fixed,
                    "cell": cell.astype(np.float32),
                    "pos_pred": pos_pred,
                },
            }
        )
    cand_env.close()
    src_env.close()
    return jobs


@torch.no_grad()
def single_point_rows(jobs: list[dict], uma, args, device: torch.device) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for chunk_start in tqdm(
        range(0, len(jobs), int(args.sp_chunk_jobs)),
        desc=f"[{args.label} shard{args.shard_idx}] UMA-SP",
        unit="chunk",
        dynamic_ncols=True,
    ):
        chunk = jobs[chunk_start:chunk_start + int(args.sp_chunk_jobs)]
        batch, ptr, batch_idx, fixed_mask, _, has_fixed, _ = make_nvbatch(chunk, device)
        copy_uma_outputs(batch, uma)
        if has_fixed:
            batch.forces[fixed_mask] = 0
        energies = batch.energy.detach().reshape(-1).float().cpu().tolist()
        from geoopt import fmax_tensor

        fvals = fmax_tensor(batch.forces, batch_idx, len(chunk), fixed_mask, has_fixed).detach().cpu().tolist()
        for i, job in enumerate(chunk):
            e = float(energies[i])
            ref = float(job["E_sys_ref"])
            out[int(job["global_i"])] = {
                "global_i": int(job["global_i"]),
                "system_key": str(job["system_key"]),
                "sample_i": int(job["sample_i"]),
                "ads_id": int(job["ads_id"]),
                "n_atoms": int(len(job["relax_input"]["numbers"])),
                "sp_E_sys": e,
                "sp_delta_E_sys": e - ref if math.isfinite(e) else float("nan"),
                "sp_fmax": float(fvals[i]),
            }
        del batch
        torch.cuda.empty_cache()
    return out


def _stats(vals: list[float]) -> dict[str, float | int | None]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def score_relaxed_validity(job: dict, result: dict) -> tuple[bool, str, str | None]:
    if not math.isfinite(float(result.get("E_sys", float("nan")))):
        return False, "uma_nonfinite", "uma_nonfinite"

    inp = job["relax_input"]
    rec = {
        "sid": int(job.get("sid", -1)),
        "system_key": str(job["system_key"]),
        "ads_id": int(job["ads_id"]),
        "pos_ref": torch.as_tensor(job["pos_ref"], dtype=torch.float32),
        "pos_pred": torch.as_tensor(result["pos_relaxed"], dtype=torch.float32),
        "pos_gt": torch.as_tensor(job["pos_gt"], dtype=torch.float32),
        "atomic_numbers": torch.as_tensor(inp["numbers"], dtype=torch.long),
        "tags": torch.as_tensor(inp["tags"], dtype=torch.long),
        "cell": torch.as_tensor(inp["cell"], dtype=torch.float32),
    }
    ar = _score_record_anomaly(rec)
    if ar.get("valid_strict"):
        status = "ok" if bool(result.get("converged")) else "uma_unconverged"
        return True, status, None
    flags = [
        k for k in ("overlap", "dissoc", "desorbed", "intercalated", "surf_changed")
        if ar.get(f"has_{k}")
    ]
    anomaly = flags[0] if flags else ar.get("error") or "anomaly"
    return False, str(anomaly), str(anomaly)


def summarize(rows: list[dict], elapsed: float, args) -> dict[str, Any]:
    n = max(len(rows), 1)
    conv = [r for r in rows if r.get("converged")]
    unconv = [r for r in rows if not r.get("converged")]
    valid = [r for r in rows if r.get("valid")]
    success = [r for r in valid if r.get("success")]
    return {
        "label": args.label,
        "mode": args.mode,
        "shard_idx": int(args.shard_idx),
        "num_shards": int(args.num_shards),
        "candidates": len(rows),
        "elapsed_sec": float(elapsed),
        "converged": len(conv),
        "unconverged": len(unconv),
        "converged_rate": len(conv) / n,
        "valid": len(valid),
        "valid_rate": len(valid) / n,
        "valid_success": len(success),
        "valid_success_rate": len(success) / max(len(valid), 1),
        "sp_delta_E_sys": _stats([r["sp_delta_E_sys"] for r in rows]),
        "valid_sp_delta_E_sys": _stats([r["sp_delta_E_sys"] for r in valid]),
        "sp_abs_delta_E_sys": _stats([abs(r["sp_delta_E_sys"]) for r in rows]),
        "valid_sp_abs_delta_E_sys": _stats([abs(r["sp_delta_E_sys"]) for r in valid]),
        "all_n_steps": _stats([r["n_steps"] for r in rows]),
        "converged_n_steps": _stats([r["n_steps"] for r in conv]),
        "valid_n_steps": _stats([r["n_steps"] for r in valid]),
        "unconverged_n_steps": _stats([r["n_steps"] for r in unconv]),
        "final_delta_E_sys": _stats([r["final_delta_E_sys"] for r in rows]),
        "valid_final_delta_E_sys": _stats([r["final_delta_E_sys"] for r in valid]),
        "final_abs_delta_E_sys": _stats([abs(r["final_delta_E_sys"]) for r in rows]),
        "valid_final_abs_delta_E_sys": _stats([abs(r["final_delta_E_sys"]) for r in valid]),
        "settings": {
            "sample_mode": args.sample_mode,
            "solver": args.solver,
            "flow_steps": args.flow_steps,
            "sde_mode": args.sde_mode,
            "sde_schedule": args.sde_schedule,
            "sde_alpha": args.sde_alpha,
            "sde_no_score": bool(args.sde_no_score),
            "si_gamma_schedule": args.si_gamma_schedule,
            "si_gamma_sigma": args.si_gamma_sigma,
            "si_epsilon_schedule": args.si_epsilon_schedule,
            "si_epsilon_scale": args.si_epsilon_scale,
            "time_schedule": args.time_schedule,
            "time_schedule_beta": args.time_schedule_beta,
            "fk_num_particles": args.fk_num_particles if args.sample_mode == "sde_fk" else None,
            "fk_lambda": args.fk_lambda if args.sample_mode == "sde_fk" else None,
            "fk_resample_interval": args.fk_resample_interval if args.sample_mode == "sde_fk" else None,
            "fk_start_time": args.fk_start_time if args.sample_mode == "sde_fk" else None,
            "fk_potential_mode": args.fk_potential_mode if args.sample_mode == "sde_fk" else None,
            "shard_by_system": bool(args.shard_by_system),
            "uma_model": args.uma_model,
            "uma_task": args.uma_task,
            "fmax": args.fmax,
            "max_steps": args.max_steps,
            "max_atoms": args.max_atoms,
            "maxstep": args.maxstep,
            "lbfgs_streaming": bool(args.lbfgs_streaming),
            "lbfgs_check_interval": args.lbfgs_check_interval,
            "lbfgs_memory": args.lbfgs_memory,
            "lbfgs_damping": args.lbfgs_damping,
            "lbfgs_alpha": args.lbfgs_alpha,
        },
    }


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["model", "random", "adsorbdiff"], required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--lmdb", default="/home1/irteam/data/processed_old/oc20dense.lmdb")
    ap.add_argument("--selected-systems", default="/home1/irteam/data/replay/mlip_pass_lbfgs_ood50/selected_ood50_systems.json")
    ap.add_argument("--cover-dir", default="/home1/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260520)
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--sample-mode", choices=["ode", "sde", "sde_fk"], default="ode")
    ap.add_argument("--solver", choices=["euler", "heun"], default="euler")
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--time-schedule", choices=["uniform", "high_t_power", "low_t_power", "beta_train"], default="uniform")
    ap.add_argument("--time-schedule-beta", type=float, default=2.0)
    ap.add_argument("--flow-batch-size", type=int, default=64)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--slab-source", default="auto", choices=["auto", "initial", "pristine_relaxed"])
    ap.add_argument("--placement-pristine-slabs", default="")
    ap.add_argument("--placement-pristine-index", default="")
    ap.add_argument("--pristine-slabs", default="/home1/irteam/data/uma_s_1p2_references/materialized/bare_slab/bare_slabs_lbfgs.pkl")
    ap.add_argument("--pristine-index", default="/home1/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense.system_index.pkl")
    ap.add_argument("--langevin-uma-model", default="uma-s-1p2")
    ap.add_argument("--langevin-uma-task", default="oc20")
    ap.add_argument("--shard-by-system", action="store_true")
    ap.add_argument("--sde-mode", choices=["atommof", "omatg_si"], default="omatg_si")
    ap.add_argument("--sde-schedule", choices=["atommof", "zero_ends"], default="atommof")
    ap.add_argument("--sde-alpha", type=float, default=1.0)
    ap.add_argument("--sde-no-score", action="store_true")
    ap.add_argument("--si-gamma-schedule", default="sqrt_t1mt")
    ap.add_argument("--si-gamma-sigma", type=float, default=0.1)
    ap.add_argument("--si-epsilon-schedule", default="vanishing_1mt")
    ap.add_argument("--si-epsilon-scale", type=float, default=1.0)
    ap.add_argument("--fk-num-particles", type=int, default=4)
    ap.add_argument("--fk-lambda", type=float, default=0.05)
    ap.add_argument("--fk-resample-interval", type=int, default=10)
    ap.add_argument("--fk-start-time", type=float, default=0.8)
    ap.add_argument("--fk-potential-mode", choices=["immediate", "difference", "max", "sum"], default="difference")
    ap.add_argument("--fk-uma-model", default="uma-s-1p2")
    ap.add_argument("--fk-uma-task", default="oc20")
    ap.add_argument("--fk-normalize-per-atom", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--adsorbdiff-metadata", default="")
    ap.add_argument("--adsorbdiff-lmdb", default="")
    ap.add_argument("--adsorbdiff-results-dir", default="")
    ap.add_argument("--uma-model", default="uma-s-1p2")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--uma-inference-settings", default="default", choices=["default", "turbo", "traineval"])
    ap.add_argument("--uma-internal-graph-version", type=int, default=0)
    ap.add_argument("--uma-execution-mode", default="")
    ap.add_argument("--uma-compile", action="store_true")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-atoms", type=int, default=32768)
    ap.add_argument("--sp-chunk-jobs", type=int, default=128)
    ap.add_argument("--maxstep", type=float, default=0.04)
    ap.add_argument("--lbfgs-memory", type=int, default=50)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    ap.add_argument("--epsilon-succ", type=float, default=0.1)
    ap.add_argument("--lbfgs-history-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-position-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-curvature-guard", choices=["abs", "positive", "ase"], default="abs")
    ap.add_argument("--lbfgs-gpu-history-guard", action="store_true")
    ap.add_argument("--lbfgs-keep-survivors-on-gpu", action="store_true")
    ap.add_argument("--lbfgs-streaming", action="store_true")
    ap.add_argument("--lbfgs-check-interval", type=int, default=20)
    ap.add_argument("--lbfgs-stream-sort", action="store_true")
    ap.add_argument("--fire-dt", type=float, default=0.1)
    ap.add_argument("--fire-dt-max", type=float, default=1.0)
    ap.add_argument("--cg-step-size", type=float, default=0.04)
    args = ap.parse_args()

    if args.mode == "model" and not args.ckpt:
        raise ValueError("--ckpt is required for --mode model")
    if args.mode == "adsorbdiff" and (not args.adsorbdiff_metadata or not args.adsorbdiff_results_dir):
        raise ValueError("--adsorbdiff-metadata and --adsorbdiff-results-dir are required")

    os.environ.setdefault("ADSGEN_ROOT", str(REPO))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    random.seed(int(args.seed) + int(args.shard_idx))
    np.random.seed(int(args.seed) + int(args.shard_idx))
    torch.manual_seed(int(args.seed) + int(args.shard_idx))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pristine_ctx = load_pristine_context(Path(args.pristine_slabs), Path(args.pristine_index))
    if pristine_ctx is None:
        raise RuntimeError(
            "strict surface-change scoring requires --pristine-slabs and --pristine-index"
        )
    refs = load_reference_by_system(Path(args.cover_dir))

    t0 = time.time()
    t_stage = time.time()
    if args.mode in ("model", "random"):
        jobs = build_model_or_random_jobs(args, device, refs)
    else:
        jobs = build_adsorbdiff_jobs(args, refs)
    generate_sec = time.time() - t_stage
    if not jobs:
        raise RuntimeError("no jobs built")

    t_stage = time.time()
    uma = load_uma(
        args.uma_model,
        args.uma_task,
        device,
        args.uma_inference_settings,
        args.uma_internal_graph_version or None,
        args.uma_execution_mode or None,
        bool(args.uma_compile) if args.uma_compile else None,
    )
    uma_load_sec = time.time() - t_stage
    t_stage = time.time()
    sp = single_point_rows(jobs, uma, args, device)
    sp_sec = time.time() - t_stage
    t_stage = time.time()
    relax_args = SimpleNamespace(**vars(args))
    relax_results = run_optimizer(jobs, uma, relax_args, device, "lbfgs", serial=False)
    relax_sec = time.time() - t_stage

    rows = []
    job_by_g = {int(j["global_i"]): j for j in jobs}
    for r in sorted(relax_results, key=lambda x: int(x["global_i"])):
        g = int(r["global_i"])
        job = job_by_g[g]
        ref = float(job["E_sys_ref"])
        row = dict(sp[g])
        valid, status, anomaly = score_relaxed_validity(job, r)
        final_delta = float(r["E_sys"] - ref) if math.isfinite(float(r["E_sys"])) else float("nan")
        row.update({
            "raw_idx": int(job["raw_idx"]),
            "E_sys_ref": ref,
            "final_E_sys": float(r["E_sys"]),
            "final_delta_E_sys": final_delta,
            "final_fmax": float(r["fmax"]),
            "n_steps": int(r["n_steps"]),
            "converged": bool(r["converged"]),
            "valid": bool(valid),
            "success": bool(valid and math.isfinite(final_delta) and final_delta <= float(args.epsilon_succ)),
            "status": status,
            "anomaly": anomaly,
            "relax_error": r.get("error"),
        })
        rows.append(row)

    elapsed = time.time() - t0
    summary = summarize(rows, elapsed, args)
    summary["stage_elapsed_sec"] = {
        "generate_or_load": float(generate_sec),
        "uma_load": float(uma_load_sec),
        "single_point": float(sp_sec),
        "relax": float(relax_sec),
        "total": float(elapsed),
    }
    summary["throughput_per_shard"] = {
        "pre_relax_candidates_per_sec": len(jobs) / generate_sec if generate_sec > 0 else None,
        "single_point_candidates_per_sec": len(jobs) / sp_sec if sp_sec > 0 else None,
        "post_relax_candidates_per_sec": len(rows) / elapsed if elapsed > 0 else None,
        "valid_post_relax_candidates_per_sec": summary["valid"] / elapsed if elapsed > 0 else None,
    }
    with (out_dir / f"shard_{args.shard_idx}.pkl").open("wb") as f:
        pickle.dump({"rows": rows, "summary": summary}, f, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / f"shard_{args.shard_idx}.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
