#!/usr/bin/env python
"""UMA batched geometry optimizer benchmark for AdsorbGen replay candidates.

This file intentionally keeps the optimizer implementations in one place:
batched FIRE, batched LBFGS, and batched nonlinear CG.  It generates identical
AdsorbGen flow predictions, validates batched output against serial
batch-size-1 execution for each optimizer, then measures batched throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def frozen_key(x):
    if isinstance(x, (list, tuple)):
        return tuple(frozen_key(v) for v in x)
    return x


def load_selected_representatives(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())

    def normalize_system_key(value):
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,)

    return [
        {
            "lmdb_id": int(r["lmdb_id"]),
            "raw_idx": int(r["raw_idx"]),
            "sid": int(r["sid"]),
            "system_key": normalize_system_key(r["system_key"]),
            "E_sys_ref": float(r["E_sys_ref"]),
        }
        for r in payload["systems"]
    ]


def read_entry(env: lmdb.Environment, idx: int) -> dict:
    with env.begin() as txn:
        raw = txn.get(str(int(idx)).encode("ascii"))
    if raw is None:
        raise KeyError(f"missing LMDB row {idx}")
    return pickle.loads(raw)


def install_adsorbgen_imports(repo: Path, adsorbates_pkl: Path) -> None:
    os.environ["ADSGEN_ROOT"] = str(repo)
    os.environ["ADSORBATES_PKL"] = str(adsorbates_pkl)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    from adsorbgen.flow import FlowConfig
    from adsorbgen.models.dit import DiTDenoiserConfig
    from adsorbgen.models.dit_v2 import DiTDenoiserV2Config
    from adsorbgen.models.factory import build_model
    import adsorbgen.models.dit as dit_mod
    import adsorbgen.models.dit_v2 as dit_v2_mod

    torch.serialization.add_safe_globals(
        [DiTDenoiserConfig, DiTDenoiserV2Config, FlowConfig]
    )
    sys.modules.setdefault("adsorbgen.model", dit_mod)
    sys.modules.setdefault("adsorbgen.model.dit", dit_mod)
    sys.modules.setdefault("adsorbgen.model.dit_v2", dit_v2_mod)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    model = build_model(hp["model_cfg"])
    state = {
        k[len("model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state, strict=False)
    model.adsorbgen_movable_mode = str(hp.get("movable_mode", "surface_ads"))
    model.adsorbgen_slab_source = str(hp.get("slab_source", "initial"))
    model.adsorbgen_pristine_slabs = str(hp.get("pristine_slabs", ""))
    model.adsorbgen_pristine_index = str(hp.get("pristine_sid_index", ""))
    model.to(device).eval()
    return model, hp["flow_cfg"]


@torch.no_grad()
def build_relax_jobs(args, device: torch.device) -> list[dict]:
    from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement
    from adsorbgen.evaluation.energy import UMAForce
    from adsorbgen.flow import euler_sample
    from adsorbgen.replay.eval import _model_cfg, _runtime_movable_mask

    selected_all = load_selected_representatives(Path(args.selected_systems))
    selected = selected_all[int(args.system_offset): int(args.system_offset) + int(args.num_systems)]
    model, flow_cfg = load_model_from_ckpt(Path(args.ckpt), device)
    model_cfg = _model_cfg(model)
    use_ads_ref = bool(getattr(model_cfg, "use_ads_ref_pos", False))
    langevin_force_model = None
    if bool(getattr(model_cfg, "use_langevin_param", False)):
        if str(getattr(model_cfg, "langevin_eval_on", "x_t")) != "x_t":
            raise ValueError("Only langevin_eval_on='x_t' is implemented")
        langevin_force_model = UMAForce(device=str(device))
    slab_source = str(getattr(model, "adsorbgen_slab_source", "initial"))
    pristine_slabs = str(getattr(model, "adsorbgen_pristine_slabs", ""))
    pristine_index = str(getattr(model, "adsorbgen_pristine_index", ""))
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

    tasks = []
    global_system_offset = int(args.system_offset)
    for sys_i, rep in enumerate(selected):
        for sample_i in range(args.num_placements):
            global_i = (global_system_offset + sys_i) * args.num_placements + sample_i
            tasks.append((global_i, sys_i, sample_i, rep))

    jobs: list[dict] = []
    for start in range(0, len(tasks), args.flow_batch_size):
        chunk = tasks[start:start + args.flow_batch_size]
        samples = []
        metas = []
        for global_i, sys_i, sample_i, rep in chunk:
            seed_i = (args.seed + int(global_i)) & 0xFFFF_FFFF
            np.random.seed(seed_i)
            random.seed(seed_i)
            sample = placement_ds[int(rep["lmdb_id"])][int(rep["raw_idx"])]
            samples.append(sample)
            metas.append((global_i, sys_i, sample_i, rep))

        batch = collate_displacement(samples)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        movable = _runtime_movable_mask(model, batch)

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
            return model(
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
            num_steps=args.flow_steps,
            use_sde=False,
            refine_final=False,
        )
        for i, (global_i, sys_i, sample_i, rep) in enumerate(metas):
            n = int(batch["pad_mask"][i].sum().item())
            cell = batch["cell"][i].detach().cpu().numpy()
            if cell.ndim == 3:
                cell = cell[0]
            source_entry = read_entry(source_envs[int(rep["lmdb_id"])], int(rep["raw_idx"]))
            pos_ref = batch["pos"][i, :n].detach().cpu().numpy().astype(np.float64)
            pos_gt = batch["pos_relaxed"][i, :n].detach().cpu().numpy().astype(np.float64)
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
                    "pos_ref": pos_ref,
                    "pos_gt": pos_gt,
                    "relax_input": {
                        "numbers": batch["atomic_numbers"][i, :n].detach().cpu().numpy().astype(np.int64),
                        "tags": batch["tags"][i, :n].detach().cpu().numpy().astype(np.int64),
                        "fixed": batch["fixed"][i, :n].detach().cpu().numpy().astype(np.int64),
                        "cell": cell.astype(np.float32),
                        "pos_pred": x_out[i, :n].detach().cpu().numpy().astype(np.float64),
                    },
                }
            )

    for env in source_envs:
        env.close()
    return jobs


def fixed_mask_from_job(job: dict) -> np.ndarray:
    p = job["relax_input"]
    fixed = np.asarray(p["fixed"], dtype=bool)
    if not fixed.any():
        fixed = np.asarray(p["tags"], dtype=int) == 0
    return fixed


def chunk_by_atom_budget(jobs: list[dict], max_atoms: int) -> list[list[dict]]:
    if max_atoms <= 0:
        return [jobs]
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_atoms = 0
    for job in jobs:
        n_atoms = int(len(job["relax_input"]["numbers"]))
        if cur and cur_atoms + n_atoms > max_atoms:
            chunks.append(cur)
            cur = []
            cur_atoms = 0
        cur.append(job)
        cur_atoms += n_atoms
    if cur:
        chunks.append(cur)
    return chunks


def make_nvbatch(jobs: list[dict], device: torch.device):
    from fast_dynamics import prepare_batch_for_dynamics
    from nvalchemi.data import AtomicData as NVAtomicData
    from nvalchemi.data import Batch as NVBatch

    pos_dtype = torch.float64 if str(getattr(make_nvbatch, "_position_dtype", "float32")) == "float64" else torch.float32
    data_list = []
    fixed_parts = []
    initial_positions = []
    local_fixed_flags = []
    for job in jobs:
        p = job["relax_input"]
        pos_src = job.get("_pos_pred_gpu")
        if isinstance(pos_src, torch.Tensor):
            pos = pos_src.detach().to(device=device, dtype=pos_dtype).clone()
        else:
            pos = torch.as_tensor(p["pos_pred"], dtype=pos_dtype, device=device)
        initial_positions.append(pos.detach().clone())
        data_list.append(
            NVAtomicData(
                positions=pos,
                atomic_numbers=torch.as_tensor(p["numbers"], dtype=torch.long, device=device),
                cell=torch.as_tensor(p["cell"], dtype=torch.float32, device=device).reshape(1, 3, 3),
                pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
            )
        )
        fixed_part = fixed_mask_from_job(job)
        fixed_parts.append(fixed_part)
        local_fixed_flags.append(bool(fixed_part.any()))
    batch = NVBatch.from_data_list(data_list)
    prepare_batch_for_dynamics(batch, outputs=("forces", "energy"))
    ptr = batch.batch_ptr.long().tolist()
    batch_idx = batch.batch_idx.to(device=device, dtype=torch.long)
    fixed_np = np.concatenate(fixed_parts) if fixed_parts else np.zeros(0, dtype=bool)
    fixed_mask = torch.as_tensor(fixed_np, dtype=torch.bool, device=device)
    return batch, ptr, batch_idx, fixed_mask, initial_positions, bool(fixed_np.any()), local_fixed_flags


def copy_uma_outputs(batch, uma) -> None:
    outputs = uma(batch)
    for key in ("forces", "energy"):
        target = getattr(batch, key)
        target.copy_(outputs[key].detach().reshape(target.shape))


def fmax_tensor(
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    n_graphs: int,
    fixed_mask: torch.Tensor,
    has_fixed: bool | None = None,
) -> torch.Tensor:
    eff = forces.detach().clone()
    if has_fixed is None:
        has_fixed = bool(fixed_mask.numel() and fixed_mask.any())
    if has_fixed:
        eff[fixed_mask] = 0
    norms = eff.norm(dim=-1)
    out = torch.zeros(n_graphs, dtype=norms.dtype, device=norms.device)
    out.scatter_reduce_(0, batch_idx, norms, reduce="amax", include_self=True)
    return out


def collect_results(jobs: list[dict], batch, ptr: list[int], fmax_values: list[float], n_steps: list[int], errors: list[str | None]) -> list[dict]:
    energies = batch.energy.detach().reshape(-1).float().cpu().tolist()
    positions = batch.positions.detach()
    out = []
    for i, job in enumerate(jobs):
        start, end = ptr[i], ptr[i + 1]
        err = errors[i]
        fmax_i = float(fmax_values[i])
        out.append(
            {
                "global_i": int(job["global_i"]),
                "n_atoms": int(len(job["relax_input"]["numbers"])),
                "converged": bool(err is None and np.isfinite(fmax_i) and fmax_i <= 0),
                "E_sys": float(energies[i]) if err is None else float("nan"),
                "fmax": fmax_i if err is None else float("nan"),
                "n_steps": int(n_steps[i]),
                "pos_relaxed": positions[start:end].cpu().numpy().astype(np.float32),
                "error": err,
            }
        )
    return out


def finalize_results(jobs, batch, ptr, batch_idx, fixed_mask, n_steps, errors, fmax_threshold, has_fixed: bool | None = None) -> list[dict]:
    fvals = fmax_tensor(batch.forces, batch_idx, len(jobs), fixed_mask, has_fixed).detach().cpu().tolist()
    out = collect_results(jobs, batch, ptr, fvals, n_steps, errors)
    for r in out:
        r["converged"] = bool(r["error"] is None and np.isfinite(r["fmax"]) and r["fmax"] <= fmax_threshold)
    return out


def lbfgs_step(pos: torch.Tensor, forces: torch.Tensor, fixed: torch.Tensor, state: dict, args, has_fixed: bool = False) -> torch.Tensor:
    hist_dtype = torch.float64 if str(getattr(args, "lbfgs_history_dtype", "float32")) == "float64" else torch.float32
    pos_flat = pos.detach().reshape(-1).to(dtype=hist_dtype).clone()
    f_eff = forces.detach().clone()
    if has_fixed:
        f_eff[fixed] = 0
    f_flat = f_eff.reshape(-1).to(dtype=hist_dtype).clone()
    if state["iteration"] > 0 and state["r0"] is not None and state["f0"] is not None:
        s0 = pos_flat - state["r0"]
        y0 = state["f0"] - f_flat
        ys = torch.dot(y0, s0)
        guard = str(getattr(args, "lbfgs_curvature_guard", "abs"))
        gpu_guard = bool(getattr(args, "lbfgs_gpu_history_guard", False))
        if guard == "ase":
            append_history = True
        elif gpu_guard:
            finite = torch.isfinite(ys)
            if guard == "positive":
                valid = finite & (ys > 1.0e-12)
            else:
                valid = finite & (torch.abs(ys) > 1.0e-12)
            s0 = torch.where(valid, s0, torch.zeros_like(s0))
            y0 = torch.where(valid, y0, torch.zeros_like(y0))
            rho0 = torch.where(valid, 1.0 / ys, torch.zeros_like(ys))
            append_history = True
        elif guard == "positive":
            append_history = bool(torch.isfinite(ys) and float(ys.item()) > 1.0e-12)
        else:
            append_history = bool(torch.isfinite(ys) and abs(float(ys.item())) > 1.0e-12)
        if append_history:
            state["s"].append(s0)
            state["y"].append(y0)
            state["rho"].append(rho0 if gpu_guard and guard != "ase" else 1.0 / ys)
            if len(state["s"]) > int(args.lbfgs_memory):
                state["s"].pop(0)
                state["y"].pop(0)
                state["rho"].pop(0)
    q = -f_flat
    alpha_hist = []
    for s_i, y_i, rho_i in reversed(list(zip(state["s"], state["y"], state["rho"]))):
        a_i = rho_i * torch.dot(s_i, q)
        alpha_hist.append(a_i)
        q = q - a_i * y_i
    z = (1.0 / float(args.lbfgs_alpha)) * q
    for (s_i, y_i, rho_i), a_i in zip(zip(state["s"], state["y"], state["rho"]), reversed(alpha_hist)):
        b_i = rho_i * torch.dot(y_i, z)
        z = z + s_i * (a_i - b_i)
    dr = (-z).reshape_as(pos)
    if has_fixed:
        dr[fixed] = 0
    max_norm = dr.norm(dim=-1).max()
    safe_norm = torch.where(torch.isfinite(max_norm), max_norm, torch.ones_like(max_norm))
    scale = torch.clamp(float(args.maxstep) / safe_norm.clamp_min(1.0e-30), max=1.0)
    scale = torch.where(torch.isfinite(max_norm), scale, torch.ones_like(scale))
    dr = dr * scale
    state["r0"] = pos_flat
    state["f0"] = f_flat
    state["iteration"] += 1
    state["n_steps"] += 1
    return (dr * float(args.lbfgs_damping)).to(dtype=pos.dtype)


def run_batched_lbfgs_chunk(jobs: list[dict], uma, args, device: torch.device) -> list[dict]:
    make_nvbatch._position_dtype = str(getattr(args, "lbfgs_position_dtype", "float32"))
    batch, ptr, batch_idx, fixed_mask, initial_positions, has_fixed, local_fixed_flags = make_nvbatch(jobs, device)
    states = [{"s": [], "y": [], "rho": [], "r0": None, "f0": None, "iteration": 0, "n_steps": 0} for _ in jobs]
    done = [False] * len(jobs)
    errors: list[str | None] = [None] * len(jobs)
    try:
        copy_uma_outputs(batch, uma)
        if has_fixed:
            batch.forces[fixed_mask] = 0
        for _ in range(int(args.max_steps)):
            converged = (fmax_tensor(batch.forces, batch_idx, len(jobs), fixed_mask, has_fixed) <= float(args.fmax)).detach().cpu().numpy()
            for i, is_converged in enumerate(converged):
                if not done[i] and bool(is_converged):
                    done[i] = True
            if all(done):
                break
            for i, state in enumerate(states):
                if done[i]:
                    continue
                start, end = ptr[i], ptr[i + 1]
                local_fixed = fixed_mask[start:end]
                local_has_fixed = local_fixed_flags[i]
                dr = lbfgs_step(batch.positions[start:end], batch.forces[start:end], local_fixed, state, args, local_has_fixed)
                batch.positions[start:end] = batch.positions[start:end] + dr
                if local_has_fixed:
                    local = batch.positions[start:end]
                    local[local_fixed] = initial_positions[i][local_fixed]
            copy_uma_outputs(batch, uma)
            if has_fixed:
                batch.forces[fixed_mask] = 0
    except Exception as exc:
        errors = [repr(exc)] * len(jobs)
    return finalize_results(jobs, batch, ptr, batch_idx, fixed_mask, [s["n_steps"] for s in states], errors, float(args.fmax), has_fixed)


def _new_lbfgs_state() -> dict:
    return {"s": [], "y": [], "rho": [], "r0": None, "f0": None, "iteration": 0, "n_steps": 0}


def _lbfgs_result_from_batch(job: dict, batch, ptr: list[int], idx: int, fmax_i: float, state: dict, err: str | None, fmax_threshold: float) -> dict:
    start, end = ptr[idx], ptr[idx + 1]
    e_sys = float(batch.energy.detach().reshape(-1)[idx].float().cpu().item()) if err is None else float("nan")
    job.pop("_pos_pred_gpu", None)
    return {
        "global_i": int(job["global_i"]),
        "n_atoms": int(len(job["relax_input"]["numbers"])),
        "converged": bool(err is None and np.isfinite(fmax_i) and fmax_i <= fmax_threshold),
        "E_sys": e_sys,
        "fmax": float(fmax_i) if err is None else float("nan"),
        "n_steps": int(state["n_steps"]),
        "pos_relaxed": batch.positions.detach()[start:end].cpu().numpy().astype(np.float32),
        "error": err,
    }


def run_streaming_lbfgs(jobs: list[dict], uma, args, device: torch.device) -> list[dict]:
    """Batched LBFGS with active-set compaction/refill every check interval.

    Candidate LBFGS histories are kept host-side in per-candidate state dicts.
    At segment boundaries we rebuild the NVBatch only from unfinished candidates,
    then refill freed atom budget with new candidates from the pool.
    """
    make_nvbatch._position_dtype = str(getattr(args, "lbfgs_position_dtype", "float32"))
    max_atoms = int(args.max_atoms)
    check_interval = max(1, int(getattr(args, "lbfgs_check_interval", 10)))
    fmax_threshold = float(args.fmax)
    max_steps = int(args.max_steps)
    pool = [
        {
            "job": job,
            "state": _new_lbfgs_state(),
            "error": None,
            "n_atoms": int(len(job["relax_input"]["numbers"])),
        }
        for job in jobs
    ]
    if bool(getattr(args, "lbfgs_stream_sort", False)):
        pool.sort(key=lambda item: item["n_atoms"])
    pool_i = 0
    active: list[dict] = []
    cur_atoms = 0
    results: list[dict] = []

    def refill() -> None:
        nonlocal pool_i, cur_atoms
        while pool_i < len(pool):
            item = pool[pool_i]
            if active and max_atoms > 0 and cur_atoms + item["n_atoms"] > max_atoms:
                break
            active.append(item)
            cur_atoms += item["n_atoms"]
            pool_i += 1
            if max_atoms > 0 and cur_atoms >= max_atoms:
                break

    refill()
    while active:
        active_jobs = [item["job"] for item in active]
        states = [item["state"] for item in active]
        errors = [item["error"] for item in active]
        terminal = [False] * len(active)
        terminal_fmax = [float("nan")] * len(active)
        try:
            batch, ptr, batch_idx, fixed_mask, initial_positions, has_fixed, local_fixed_flags = make_nvbatch(active_jobs, device)
            copy_uma_outputs(batch, uma)
            if has_fixed:
                batch.forces[fixed_mask] = 0
            for _ in range(check_interval):
                if all(int(state["n_steps"]) >= max_steps for state in states):
                    break
                any_updated = False
                for i, state in enumerate(states):
                    if int(state["n_steps"]) >= max_steps:
                        continue
                    start, end = ptr[i], ptr[i + 1]
                    local_fixed = fixed_mask[start:end]
                    local_has_fixed = local_fixed_flags[i]
                    dr = lbfgs_step(batch.positions[start:end], batch.forces[start:end], local_fixed, state, args, local_has_fixed)
                    batch.positions[start:end] = batch.positions[start:end] + dr
                    if local_has_fixed:
                        local = batch.positions[start:end]
                        local[local_fixed] = initial_positions[i][local_fixed]
                    any_updated = True
                if not any_updated:
                    break
                copy_uma_outputs(batch, uma)
                if has_fixed:
                    batch.forces[fixed_mask] = 0

            fvals = fmax_tensor(batch.forces, batch_idx, len(active_jobs), fixed_mask, has_fixed).detach().cpu().tolist()
            for i, fmax_i in enumerate(fvals):
                if not terminal[i] and (
                    (np.isfinite(fmax_i) and float(fmax_i) <= fmax_threshold)
                    or int(states[i]["n_steps"]) >= max_steps
                ):
                    terminal[i] = True
                    terminal_fmax[i] = float(fmax_i)
                elif terminal[i] and not np.isfinite(terminal_fmax[i]):
                    terminal_fmax[i] = float(fmax_i)

            survivors: list[dict] = []
            cur_atoms = 0
            for i, item in enumerate(active):
                start, end = ptr[i], ptr[i + 1]
                if terminal[i]:
                    results.append(_lbfgs_result_from_batch(
                        item["job"], batch, ptr, i, terminal_fmax[i], item["state"], item["error"], fmax_threshold
                    ))
                else:
                    if bool(getattr(args, "lbfgs_keep_survivors_on_gpu", False)):
                        item["job"]["_pos_pred_gpu"] = batch.positions.detach()[start:end].clone()
                    else:
                        item["job"]["relax_input"]["pos_pred"] = batch.positions.detach()[start:end].cpu().numpy().astype(np.float64)
                    survivors.append(item)
                    cur_atoms += item["n_atoms"]
            active = survivors
            refill()
        except Exception as exc:
            err = repr(exc)
            for item in active:
                item["error"] = err
            # Preserve old behavior: a batch-level failure marks all active jobs.
            failed_jobs = [item["job"] for item in active]
            batch, ptr, batch_idx, fixed_mask, _, _, _ = make_nvbatch(failed_jobs, device)
            for i, item in enumerate(active):
                results.append(_lbfgs_result_from_batch(item["job"], batch, ptr, i, float("nan"), item["state"], err, fmax_threshold))
            active = []
    results.sort(key=lambda r: int(r["global_i"]))
    return results


def run_batched_cg_chunk(jobs: list[dict], uma, args, device: torch.device) -> list[dict]:
    batch, ptr, batch_idx, fixed_mask, initial_positions, has_fixed, local_fixed_flags = make_nvbatch(jobs, device)
    states = [{"direction": None, "prev_force": None, "n_steps": 0} for _ in jobs]
    done = [False] * len(jobs)
    errors: list[str | None] = [None] * len(jobs)
    try:
        copy_uma_outputs(batch, uma)
        if has_fixed:
            batch.forces[fixed_mask] = 0
        for _ in range(int(args.max_steps)):
            converged = (fmax_tensor(batch.forces, batch_idx, len(jobs), fixed_mask, has_fixed) <= float(args.fmax)).detach().cpu().numpy()
            for i, is_converged in enumerate(converged):
                if not done[i] and bool(is_converged):
                    done[i] = True
            if all(done):
                break
            for i, state in enumerate(states):
                if done[i]:
                    continue
                start, end = ptr[i], ptr[i + 1]
                local_fixed = fixed_mask[start:end]
                local_has_fixed = local_fixed_flags[i]
                force = batch.forces[start:end].detach().clone()
                if local_has_fixed:
                    force[local_fixed] = 0
                f_flat = force.reshape(-1)
                if state["direction"] is None or state["prev_force"] is None:
                    direction = f_flat
                else:
                    prev = state["prev_force"]
                    denom = torch.dot(prev, prev)
                    beta = torch.tensor(0.0, dtype=f_flat.dtype, device=f_flat.device)
                    if torch.isfinite(denom) and float(denom.item()) > 1.0e-20:
                        beta = torch.clamp(torch.dot(f_flat, f_flat - prev) / denom, min=0.0)
                    direction = f_flat + beta * state["direction"]
                    if torch.dot(direction, f_flat) <= 0:
                        direction = f_flat
                dr = direction.reshape_as(force) * float(args.cg_step_size)
                if local_has_fixed:
                    dr[local_fixed] = 0
                max_norm = dr.norm(dim=-1).max()
                if torch.isfinite(max_norm) and float(max_norm.item()) >= float(args.maxstep):
                    dr = dr * (float(args.maxstep) / max_norm)
                batch.positions[start:end] = batch.positions[start:end] + dr
                if local_has_fixed:
                    local = batch.positions[start:end]
                    local[local_fixed] = initial_positions[i][local_fixed]
                state["direction"] = direction.detach().clone()
                state["prev_force"] = f_flat.detach().clone()
                state["n_steps"] += 1
            copy_uma_outputs(batch, uma)
            if has_fixed:
                batch.forces[fixed_mask] = 0
    except Exception as exc:
        errors = [repr(exc)] * len(jobs)
    return finalize_results(jobs, batch, ptr, batch_idx, fixed_mask, [s["n_steps"] for s in states], errors, float(args.fmax), has_fixed)


def run_batched_fire_chunk(jobs: list[dict], uma, args, device: torch.device) -> list[dict]:
    batch, ptr, batch_idx, fixed_mask, initial_positions, has_fixed, local_fixed_flags = make_nvbatch(jobs, device)
    states = [
        {
            "vel": torch.zeros((ptr[i + 1] - ptr[i], 3), dtype=torch.float32, device=device),
            "dt": float(args.fire_dt),
            "alpha": 0.1,
            "n_pos": 0,
            "n_steps": 0,
        }
        for i in range(len(jobs))
    ]
    done = [False] * len(jobs)
    errors: list[str | None] = [None] * len(jobs)
    try:
        if has_fixed:
            batch.positions[fixed_mask] = batch.positions[fixed_mask]
        copy_uma_outputs(batch, uma)
        if has_fixed:
            batch.forces[fixed_mask] = 0
        for _ in range(int(args.max_steps)):
            converged = (fmax_tensor(batch.forces, batch_idx, len(jobs), fixed_mask, has_fixed) <= float(args.fmax)).detach().cpu().numpy()
            for i, is_converged in enumerate(converged):
                if not done[i] and bool(is_converged):
                    done[i] = True
            if all(done):
                break
            for i, state in enumerate(states):
                if done[i]:
                    continue
                start, end = ptr[i], ptr[i + 1]
                local_fixed = fixed_mask[start:end]
                local_has_fixed = local_fixed_flags[i]
                force = batch.forces[start:end].detach().clone()
                if local_has_fixed:
                    force[local_fixed] = 0
                vel = state["vel"]
                vf = torch.sum(vel * force)
                ff = torch.sum(force * force)
                vv = torch.sum(vel * vel)
                if vf > 0.0 and ff > 0.0 and vv > 0.0:
                    vel = (1.0 - state["alpha"]) * vel + state["alpha"] * force / torch.sqrt(ff) * torch.sqrt(vv)
                    if state["n_pos"] > 5:
                        state["dt"] = min(state["dt"] * 1.1, float(args.fire_dt_max))
                        state["alpha"] *= 0.99
                    state["n_pos"] += 1
                else:
                    vel = torch.zeros_like(vel)
                    state["alpha"] = 0.1
                    state["dt"] *= 0.5
                    state["n_pos"] = 0
                vel = vel + state["dt"] * force
                dr = state["dt"] * vel
                if local_has_fixed:
                    dr[local_fixed] = 0
                    vel[local_fixed] = 0
                normdr = torch.sqrt(torch.sum(dr * dr))
                if torch.isfinite(normdr) and float(normdr.item()) > float(args.maxstep):
                    dr = dr * (float(args.maxstep) / normdr)
                batch.positions[start:end] = batch.positions[start:end] + dr
                if local_has_fixed:
                    local = batch.positions[start:end]
                    local[local_fixed] = initial_positions[i][local_fixed]
                state["vel"] = vel.detach().clone()
                state["n_steps"] += 1
            copy_uma_outputs(batch, uma)
            if has_fixed:
                batch.forces[fixed_mask] = 0
    except Exception as exc:
        errors = [repr(exc)] * len(jobs)
    return finalize_results(jobs, batch, ptr, batch_idx, fixed_mask, [s["n_steps"] for s in states], errors, float(args.fmax), has_fixed)


def run_optimizer(jobs: list[dict], uma, args, device: torch.device, algorithm: str, serial: bool) -> list[dict]:
    runners = {
        "fire": run_batched_fire_chunk,
        "lbfgs": run_batched_lbfgs_chunk,
        "cg": run_batched_cg_chunk,
    }
    if algorithm == "lbfgs" and bool(getattr(args, "lbfgs_streaming", False)) and not serial:
        return run_streaming_lbfgs(jobs, uma, args, device)
    runner = runners[algorithm]
    out = []
    chunks = [[job] for job in jobs] if serial else chunk_by_atom_budget(jobs, int(args.max_atoms))
    for chunk in chunks:
        out.extend(runner(chunk, uma, args, device))
    return out


def compare_results(serial: list[dict], batched: list[dict]) -> dict:
    by_batch = {r["global_i"]: r for r in batched}
    rows = []
    for s in serial:
        b = by_batch[s["global_i"]]
        dpos = np.asarray(b["pos_relaxed"], dtype=np.float64) - np.asarray(s["pos_relaxed"], dtype=np.float64)
        rows.append(
            {
                "global_i": s["global_i"],
                "serial_converged": bool(s["converged"]),
                "batched_converged": bool(b["converged"]),
                "serial_E": float(s["E_sys"]),
                "batched_E": float(b["E_sys"]),
                "dE": float(b["E_sys"] - s["E_sys"]),
                "serial_fmax": float(s["fmax"]),
                "batched_fmax": float(b["fmax"]),
                "dfmax": float(b["fmax"] - s["fmax"]),
                "pos_rmse": float(np.sqrt(np.mean(dpos * dpos))),
                "pos_max_abs": float(np.max(np.abs(dpos))),
            }
        )
    finite = [r for r in rows if np.isfinite(r["dE"]) and np.isfinite(r["pos_rmse"])]
    both_conv = [r for r in finite if r["serial_converged"] and r["batched_converged"]]

    def mean_abs(key: str, subset: list[dict]) -> float | None:
        return float(np.mean([abs(r[key]) for r in subset])) if subset else None

    def max_abs(key: str, subset: list[dict]) -> float | None:
        return float(np.max([abs(r[key]) for r in subset])) if subset else None

    return {
        "rows": rows,
        "converged_agreement": sum(1 for r in rows if r["serial_converged"] == r["batched_converged"]),
        "serial_converged": sum(1 for r in rows if r["serial_converged"]),
        "batched_converged": sum(1 for r in rows if r["batched_converged"]),
        "mean_abs_dE": mean_abs("dE", finite),
        "max_abs_dE": max_abs("dE", finite),
        "mean_pos_rmse": float(np.mean([r["pos_rmse"] for r in finite])) if finite else None,
        "max_pos_rmse": max_abs("pos_rmse", finite),
        "both_converged": len(both_conv),
        "both_converged_max_abs_dE": max_abs("dE", both_conv),
        "both_converged_max_pos_rmse": max_abs("pos_rmse", both_conv),
    }


def summarize_throughput(results: list[dict], elapsed: float) -> dict:
    return {
        "elapsed_sec": float(elapsed),
        "candidates": len(results),
        "candidates_per_sec": len(results) / elapsed if elapsed > 0 else None,
        "converged": sum(1 for r in results if r["converged"]),
        "converged_rate": sum(1 for r in results if r["converged"]) / max(len(results), 1),
    }


def load_uma(
    model_name: str,
    task: str,
    device: torch.device,
    inference_settings: str = "default",
    internal_graph_version: int | None = None,
    execution_mode: str | None = None,
    compile_model: bool | None = None,
):
    from dataclasses import replace
    import sys
    import types

    if internal_graph_version is not None and int(internal_graph_version) == 3:
        from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
        from nvalchemiops.torch.neighbors import neighbor_list

        compat_name = "nvalchemiops.neighborlist.neighborlist"
        compat_mod = types.ModuleType(compat_name)
        compat_mod.neighbor_list = neighbor_list
        sys.modules[compat_name] = compat_mod

        rg_mod = sys.modules.get("fairchem.core.graph.radius_graph_pbc_nvidia")
        if rg_mod is not None:
            rg_mod.estimate_max_neighbors = estimate_max_neighbors
            rg_mod.neighbor_list = neighbor_list

    from fast_dynamics import UMAWrapper
    from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings

    settings = guess_inference_settings(inference_settings)
    updates = {}
    if internal_graph_version is not None and int(internal_graph_version) > 0:
        updates["internal_graph_gen_version"] = int(internal_graph_version)
    if execution_mode:
        updates["execution_mode"] = str(execution_mode)
    if compile_model is not None:
        updates["compile"] = bool(compile_model)
    if updates:
        settings = replace(settings, **updates)
    return UMAWrapper.from_checkpoint(model_name, task_name=task, device=device, inference_settings=settings)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/minkyu/Cat-bench/AdsorbGen")
    ap.add_argument("--adsorbates-pkl", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-lmdb", nargs="+", required=True)
    ap.add_argument("--selected-systems", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--algorithms", nargs="+", default=["fire", "lbfgs", "cg"], choices=["fire", "lbfgs", "cg"])
    ap.add_argument("--seed", type=int, default=20260526)
    ap.add_argument("--num-systems", type=int, default=32)
    ap.add_argument("--system-offset", type=int, default=0)
    ap.add_argument("--num-placements", type=int, default=1)
    ap.add_argument("--accuracy-systems", type=int, default=8)
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--flow-batch-size", type=int, default=32)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--uma-model", default="uma-s-1p2")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--uma-inference-settings", default="default", choices=["default", "turbo", "traineval"])
    ap.add_argument("--uma-internal-graph-version", type=int, default=0)
    ap.add_argument("--uma-execution-mode", default="")
    ap.add_argument("--uma-compile", action="store_true")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-atoms", type=int, default=4096)
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
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    install_adsorbgen_imports(repo, Path(args.adsorbates_pkl).resolve())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")

    t0 = time.time()
    jobs = build_relax_jobs(args, device)
    flow_elapsed = time.time() - t0
    accuracy_jobs = jobs[: min(len(jobs), int(args.accuracy_systems) * int(args.num_placements))]

    report: dict[str, Any] = {
        "settings": vars(args),
        "flow_elapsed_sec": flow_elapsed,
        "num_jobs": len(jobs),
        "accuracy_jobs": len(accuracy_jobs),
        "algorithms": {},
    }

    for algorithm in args.algorithms:
        uma = load_uma(
            args.uma_model,
            args.uma_task,
            device,
            args.uma_inference_settings,
            args.uma_internal_graph_version or None,
            args.uma_execution_mode or None,
            bool(args.uma_compile) if args.uma_compile else None,
        )
        t_serial = time.time()
        serial_results = run_optimizer(accuracy_jobs, uma, args, device, algorithm, serial=True)
        serial_elapsed = time.time() - t_serial
        t_batch_acc = time.time()
        batch_acc_results = run_optimizer(accuracy_jobs, uma, args, device, algorithm, serial=False)
        batch_acc_elapsed = time.time() - t_batch_acc
        acc = compare_results(serial_results, batch_acc_results)
        t_batch = time.time()
        batch_results = run_optimizer(jobs, uma, args, device, algorithm, serial=False)
        batch_elapsed = time.time() - t_batch
        report["algorithms"][algorithm] = {
            "accuracy": acc,
            "serial_accuracy_throughput": summarize_throughput(serial_results, serial_elapsed),
            "batched_accuracy_throughput": summarize_throughput(batch_acc_results, batch_acc_elapsed),
            "throughput": summarize_throughput(batch_results, batch_elapsed),
        }
        del uma
        torch.cuda.empty_cache()

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
