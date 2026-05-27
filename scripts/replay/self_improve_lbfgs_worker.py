#!/usr/bin/env python
"""Self-improvement replay with model sampling + ASE L-BFGS UMA relaxation.

This worker owns a modulo shard of a fixed draw of unique train/val-ID systems.
For each selected system it generates N placements, relaxes each candidate with
ASE LBFGS, and records candidates whose UMA E_sys is lower than the existing
train/val-ID system minimum. Success entries include relaxed coordinates so a
later materialization step can replace the original training target per system.
Optionally, the worker can also persist valid near-minimum candidates for
moving-window replay buffers without changing the success statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from tqdm.auto import tqdm

REPO = Path(os.environ.get("ADSGEN_ROOT", "/home/irteam/AdsorbGen"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement  # noqa: E402
from adsorbgen.evaluation.metrics import load_pristine_context, _score_record_anomaly  # noqa: E402
from adsorbgen.flow import FlowConfig, FKSteeringConfig, euler_sample  # noqa: E402
from adsorbgen.models.dit import DiTDenoiserConfig  # noqa: E402
from adsorbgen.models.dit_v2 import DiTDenoiserV2Config  # noqa: E402
from adsorbgen.models.factory import build_model  # noqa: E402
from adsorbgen.replay.eval import _model_cfg, _runtime_movable_mask  # noqa: E402


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    torch.serialization.add_safe_globals(
        [DiTDenoiserConfig, DiTDenoiserV2Config, FlowConfig]
    )
    import sys
    import adsorbgen.models.dit as _dit_mod
    import adsorbgen.models.dit_v2 as _dit_v2_mod
    sys.modules.setdefault("adsorbgen.model", _dit_mod)
    sys.modules.setdefault("adsorbgen.model.dit", _dit_mod)
    sys.modules.setdefault("adsorbgen.model.dit_v2", _dit_v2_mod)
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    model = build_model(hp["model_cfg"])
    sd = ck["state_dict"]
    stripped = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(stripped, strict=False)
    model.adsorbgen_movable_mode = str(hp.get("movable_mode", "surface_ads"))
    model.to(device).eval()
    return model, hp["flow_cfg"]


def lmdb_length(txn) -> int:
    raw = txn.get(b"length")
    return int(pickle.loads(raw)) if raw is not None else int(txn.stat()["entries"])


def read_entry(env: lmdb.Environment, idx: int) -> dict:
    with env.begin() as txn:
        raw = txn.get(str(int(idx)).encode("ascii"))
    if raw is None:
        raise KeyError(f"missing LMDB row {idx}")
    return pickle.loads(raw)


def entry_reference_energy(entry: dict, gt_info: dict | None) -> float | None:
    """Energy used as the current per-row reference for self-improvement.

    Moving-window LMDBs carry replay-derived energies in metadata.  Plain
    processed LMDBs fall back to the static gt_index per-row E_sys_own.
    """

    for key in (
        "self_improve_window_E_sys",
        "self_improve_E_sys",
        "mlip_e_total",
    ):
        if key in entry and entry[key] is not None:
            try:
                value = float(entry[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
    if isinstance(gt_info, dict) and gt_info.get("E_sys_own") is not None:
        value = float(gt_info["E_sys_own"])
        if np.isfinite(value):
            return value
    return None


def build_unique_representatives(lmdb_paths: list[str], gt_index_by_sid: dict) -> list[dict]:
    best_by_system: dict[tuple, dict] = {}
    for lid, path in enumerate(lmdb_paths):
        env = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            n = lmdb_length(txn)
            mask_raw = txn.get(b"anomaly_mask")
            mask = np.asarray(pickle.loads(mask_raw), dtype=np.int8)[:n] if mask_raw else None
            kept_rows = 0
            for raw_idx in range(n):
                if mask is not None and int(mask[raw_idx]) != 0:
                    continue
                raw = txn.get(str(raw_idx).encode("ascii"))
                if raw is None:
                    continue
                entry = pickle.loads(raw)
                sid = int(entry.get("sid", -1))
                gi = gt_index_by_sid.get(sid)
                if not (
                    isinstance(gi, dict)
                    and gi.get("eligible")
                    and gi.get("system_key") is not None
                ):
                    continue
                e_ref = entry_reference_energy(entry, gi)
                if e_ref is None:
                    continue
                sk = frozen_key(gi["system_key"])
                cur = best_by_system.get(sk)
                if cur is None or float(e_ref) < float(cur["E_sys_ref"]):
                    best_by_system[sk] = {
                        "lmdb_id": lid,
                        "raw_idx": int(raw_idx),
                        "sid": sid,
                        "system_key": sk,
                        "E_sys_ref": float(e_ref),
                    }
                kept_rows += 1
        env.close()
        print(
            f"[index] {Path(path).name}: kept {kept_rows} eligible rows; "
            f"unique systems so far={len(best_by_system)}",
            flush=True,
        )
    reps = list(best_by_system.values())
    reps.sort(key=lambda r: str(r["system_key"]))
    return reps


def load_selected_representatives(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    out = []
    for r in payload["systems"]:
        out.append({
            "lmdb_id": int(r["lmdb_id"]),
            "raw_idx": int(r["raw_idx"]),
            "sid": int(r["sid"]),
            "system_key": tuple(r["system_key"]),
            "E_sys_ref": float(r["E_sys_ref"]),
        })
    return out


def frozen_key(x):
    if isinstance(x, (list, tuple)):
        return tuple(frozen_key(v) for v in x)
    return x


def fixed_atoms_from_prediction(p: dict) -> Atoms:
    atoms = Atoms(
        numbers=np.asarray(p["numbers"], dtype=int),
        positions=np.asarray(p["pos_pred"], dtype=float),
        cell=np.asarray(p["cell"], dtype=float),
        pbc=True,
        tags=np.asarray(p["tags"], dtype=int).tolist(),
    )
    fixed = np.asarray(p["fixed"], dtype=bool)
    if not fixed.any():
        fixed = np.asarray(p["tags"], dtype=int) == 0
    if fixed.any():
        atoms.set_constraint(FixAtoms(indices=np.where(fixed)[0].tolist()))
    return atoms


def relax_lbfgs_one(p: dict, calc, args) -> dict:
    atoms = fixed_atoms_from_prediction(p)
    atoms.calc = calc
    opt = LBFGS(
        atoms,
        logfile=None,
        maxstep=args.lbfgs_maxstep,
        memory=args.lbfgs_memory,
        damping=args.lbfgs_damping,
        alpha=args.lbfgs_alpha,
    )
    try:
        converged = bool(opt.run(fmax=args.uma_fmax, steps=args.uma_max_steps))
        e_sys = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        fmax = float(np.max(np.linalg.norm(forces, axis=1)))
        relaxed_pos = atoms.get_positions().astype(np.float32)
        err = None
    except Exception as exc:
        converged = False
        e_sys = float("nan")
        fmax = float("nan")
        relaxed_pos = np.asarray(p["pos_pred"], dtype=np.float32)
        err = repr(exc)
    return {
        "converged": converged,
        "E_sys": e_sys,
        "fmax": fmax,
        "n_steps": int(getattr(opt, "nsteps", 0)),
        "pos_relaxed": relaxed_pos,
        "error": err,
    }


def relax_lbfgs_batch(relax_jobs: list[dict], calc, args, *, batcher=None, calculator_cls=None) -> list[dict]:
    """Relax a list of candidates, optionally batching UMA inference requests.

    The ASE calculator keeps mutable state, so concurrent mode creates one
    calculator per worker thread while sharing only the fairchem batch predictor.
    """

    if not relax_jobs:
        return []
    if batcher is None:
        return [relax_lbfgs_one(job["relax_input"], calc, args) for job in relax_jobs]
    if calculator_cls is None:
        raise ValueError("calculator_cls is required when using fairchem batcher")

    tls = threading.local()

    def thread_calc():
        local_calc = getattr(tls, "calc", None)
        if local_calc is None:
            local_calc = calculator_cls(batcher.batch_predict_unit, task_name=args.uma_task)
            tls.calc = local_calc
        return local_calc

    def run_one(job_idx: int, job: dict) -> tuple[int, dict]:
        return job_idx, relax_lbfgs_one(job["relax_input"], thread_calc(), args)

    out: list[dict | None] = [None] * len(relax_jobs)
    futures = [
        batcher.executor.submit(run_one, job_idx, job)
        for job_idx, job in enumerate(relax_jobs)
    ]
    for fut in as_completed(futures):
        job_idx, result = fut.result()
        out[job_idx] = result
    return [r for r in out if r is not None]


def status_from_anomaly(ar: dict) -> tuple[bool, str | None]:
    if ar.get("valid_strict"):
        return True, None
    flags = [
        k for k in ("overlap", "dissoc", "desorbed", "intercalated", "surf_changed")
        if ar.get(f"has_{k}")
    ]
    return False, flags[0] if flags else ar.get("error") or "anomaly"


def summarize_rows(rows: list[dict], selected_systems: list[tuple]) -> dict:
    by_sys: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_sys[frozen_key(r["system_key"])].append(r)
    n_sys_with_success = sum(
        1 for sk in selected_systems if any(r.get("success") for r in by_sys.get(frozen_key(sk), []))
    )
    total = max(len(rows), 1)
    return {
        "systems": len(selected_systems),
        "candidates": len(rows),
        "systems_with_success": int(n_sys_with_success),
        "converged": sum(1 for r in rows if r.get("converged")),
        "valid": sum(1 for r in rows if r.get("valid")),
        "success": sum(1 for r in rows if r.get("success")),
        "converged_rate": sum(1 for r in rows if r.get("converged")) / total,
        "valid_rate": sum(1 for r in rows if r.get("valid")) / total,
        "success_sample_rate": sum(1 for r in rows if r.get("success")) / total,
        "success_system_rate": n_sys_with_success / max(len(selected_systems), 1),
    }


def write_progress(path: Path, args, rows: list[dict], successes: list[dict], selected: list[tuple], t0: float) -> None:
    payload = summarize_rows(rows, selected)
    payload.update({
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
        "target_candidates": len(selected) * args.num_placements,
        "elapsed_sec": time.time() - t0,
        "success_entries": len(successes),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    atomic_write_json(path, payload)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-lmdb", nargs="+", required=True)
    ap.add_argument("--gt-index", default="/home/irteam/data/replay/gt_index_by_sid_oc20_lbfgs.pkl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260523)
    ap.add_argument("--num-systems", type=int, default=10000)
    ap.add_argument("--num-placements", type=int, default=10)
    ap.add_argument("--selected-systems", default="")
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--flow-batch-size", type=int, default=32)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--use-sde", action="store_true",
                    help="use AtomMOF-style SDE sampling")
    ap.add_argument("--refine-final", action="store_true")
    ap.add_argument("--sde-schedule", type=str, default="atommof",
                    choices=["atommof", "zero_ends"],
                    help="g²(t): atommof=0.5(1-t), zero_ends=α·t(1-t)")
    ap.add_argument("--sde-alpha", type=float, default=1.0,
                    help="multiplier for zero_ends schedule")
    # FK steering (Feynman-Kac particle resampling guided by UMA energy)
    ap.add_argument("--fk-particles", type=int, default=0,
                    help="P: FK particles per (system, placement). 0 disables FK.")
    ap.add_argument("--fk-lambda", type=float, default=2.0)
    ap.add_argument("--fk-resample-interval", type=int, default=5)
    ap.add_argument("--fk-start-time", type=float, default=0.80)
    ap.add_argument("--fk-potential", type=str, default="immediate",
                    choices=["immediate", "difference", "max", "sum"])
    ap.add_argument("--fk-energy", type=str, default="uma",
                    choices=["zero", "uma"])
    ap.add_argument("--fk-uma-model", type=str, default="uma-s-1p1")
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--uma-fmax", type=float, default=0.05)
    ap.add_argument("--uma-max-steps", type=int, default=300)
    ap.add_argument("--lbfgs-maxstep", type=float, default=0.04)
    ap.add_argument("--lbfgs-memory", type=int, default=50)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    ap.add_argument(
        "--use-fairchem-batcher",
        action="store_true",
        help="batch concurrent ASE L-BFGS UMA inference requests via fairchem InferenceBatcher",
    )
    ap.add_argument(
        "--lbfgs-concurrency",
        type=int,
        default=1,
        help="number of concurrent ASE L-BFGS jobs inside this worker when batcher is enabled",
    )
    ap.add_argument(
        "--batcher-max-atoms",
        type=int,
        default=4096,
        help="fairchem batcher max atom budget, not number of structures",
    )
    ap.add_argument(
        "--batcher-wait-timeout-s",
        type=float,
        default=0.02,
        help="fairchem batcher wait timeout for collecting concurrent inference requests",
    )
    ap.add_argument("--success-margin", type=float, default=0.0)
    ap.add_argument(
        "--save-window-candidates",
        action="store_true",
        help=(
            "also persist valid/converged candidates with relaxed coordinates "
            "for moving-window buffer materialization"
        ),
    )
    ap.add_argument(
        "--candidate-window-ev",
        type=float,
        default=0.1,
        help=(
            "save window candidates with E_sys <= current E_sys_ref + this value; "
            "success metrics are unchanged"
        ),
    )
    ap.add_argument("--pristine-slabs", default="/home/irteam/results/pristine_slabs/is2res.pkl")
    ap.add_argument("--pristine-index", default="/home/irteam/results/pristine_slabs/is2res.sid_index.pkl")
    ap.add_argument("--progress-every", type=int, default=32)
    args = ap.parse_args()

    # FK steering needs SDE for particles to diverge; auto-enable like AtomMOF.
    if args.fk_particles > 0 and not args.use_sde:
        print("[fk] WARNING: --fk-particles>0 but --use-sde not set. "
              "Auto-enabling --use-sde.", flush=True)
        args.use_sde = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    progress_path = logs_dir / f"progress_shard{args.shard_idx}.json"

    random.seed(args.seed + args.shard_idx)
    np.random.seed(args.seed + args.shard_idx)
    torch.manual_seed(args.seed + args.shard_idx)

    selected_path = Path(args.selected_systems) if args.selected_systems else out_dir / "selected_systems.json"
    if selected_path.exists():
        selected_reps = load_selected_representatives(selected_path)
        if len(selected_reps) < args.num_systems:
            raise RuntimeError(f"{selected_path} has only {len(selected_reps)} systems")
        selected_reps = selected_reps[: args.num_systems]
    else:
        with open(args.gt_index, "rb") as f:
            gt_index_by_sid = pickle.load(f)
        reps = build_unique_representatives(args.train_lmdb, gt_index_by_sid)
        if len(reps) < args.num_systems:
            raise RuntimeError(f"only {len(reps)} eligible unique systems, need {args.num_systems}")
        rng = np.random.default_rng(args.seed)
        selected_idx = sorted(rng.choice(len(reps), size=args.num_systems, replace=False).tolist())
        selected_reps = [reps[i] for i in selected_idx]
        if args.shard_idx == 0:
            atomic_write_json(selected_path, {
                "seed": args.seed,
                "num_systems": args.num_systems,
                "num_placements": args.num_placements,
                "train_lmdb": args.train_lmdb,
                "systems": [
                    {
                        "system_key": list(r["system_key"]),
                        "lmdb_id": int(r["lmdb_id"]),
                        "raw_idx": int(r["raw_idx"]),
                        "sid": int(r["sid"]),
                        "E_sys_ref": float(r["E_sys_ref"]),
                    }
                    for r in selected_reps
                ],
            })
    selected_systems = [tuple(r["system_key"]) for r in selected_reps]

    tasks = []
    for sys_i, rep in enumerate(selected_reps):
        for sample_i in range(args.num_placements):
            global_i = sys_i * args.num_placements + sample_i
            if global_i % args.num_shards == args.shard_idx:
                tasks.append((global_i, sys_i, sample_i, rep))

    print(
        f"[shard{args.shard_idx}] selected_systems={len(selected_reps)} "
        f"tasks={len(tasks)}/{len(selected_reps) * args.num_placements}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    model, flow_cfg = load_model_from_ckpt(Path(args.ckpt), device)
    use_ads_ref = bool(getattr(_model_cfg(model), "use_ads_ref_pos", False))

    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    predict_unit = pretrained_mlip.get_predict_unit(args.uma_model, device=str(device))
    batcher = None
    if args.use_fairchem_batcher:
        if args.lbfgs_concurrency < 1:
            raise ValueError("--lbfgs-concurrency must be >= 1")
        from fairchem.core.calculate._batch import InferenceBatcher

        batcher = InferenceBatcher(
            predict_unit,
            max_batch_size=args.batcher_max_atoms,
            batch_wait_timeout_s=args.batcher_wait_timeout_s,
            concurrency_backend_options={"max_workers": int(args.lbfgs_concurrency)},
        )
        calc = None
        print(
            "[batcher] enabled "
            f"concurrency={args.lbfgs_concurrency} "
            f"max_atoms={args.batcher_max_atoms} "
            f"timeout_s={args.batcher_wait_timeout_s}",
            flush=True,
        )
    else:
        calc = FAIRChemCalculator(predict_unit, task_name=args.uma_task)
    load_pristine_context(Path(args.pristine_slabs), Path(args.pristine_index))

    # FK steering energy model (lazy-init only when FK is active and uma is requested).
    fk_energy_model = None
    if args.fk_particles > 0:
        if args.fk_energy == "uma":
            from adsorbgen.evaluation.energy import UMAEnergy
            fk_energy_model = UMAEnergy(
                model_name=args.fk_uma_model,
                device=str(device),
                task_name=args.uma_task,
                normalize_per_atom=True,
            )
            print(f"[fk] energy_model: UMAEnergy({args.fk_uma_model})", flush=True)
        else:
            # 'zero' uniform potential — useful for FK pipeline debug.
            class _ZeroEnergy(torch.nn.Module):
                def forward(self, pos, cell, anum, pad):
                    return torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
            fk_energy_model = _ZeroEnergy().to(device)

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

    rows: list[dict] = []
    success_entries: list[dict] = []
    window_candidate_entries: list[dict] = []
    t0 = time.time()
    write_progress(progress_path, args, rows, success_entries, selected_systems, t0)

    for start in tqdm(
        range(0, len(tasks), args.flow_batch_size),
        desc=f"[shard{args.shard_idx}] flow+lbfgs",
        unit="batch",
        dynamic_ncols=True,
    ):
        chunk_tasks = tasks[start:start + args.flow_batch_size]
        samples = []
        metas = []
        for global_i, sys_i, sample_i, rep in chunk_tasks:
            seed_i = (args.seed + int(global_i)) & 0xFFFF_FFFF
            np.random.seed(seed_i)
            random.seed(seed_i)
            sample = placement_ds[int(rep["lmdb_id"])][int(rep["raw_idx"])]
            samples.append(sample)
            metas.append((global_i, sys_i, sample_i, rep, read_entry(source_envs[int(rep["lmdb_id"])], int(rep["raw_idx"]))))

        batch = collate_displacement(samples)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        movable = _runtime_movable_mask(model, batch)
        BK = batch["pos"].shape[0]

        # Replicate batch P times for FK particles (P=1 is identity).
        P = max(args.fk_particles, 1)
        if P > 1:
            work = {k: (v.repeat_interleave(P, dim=0) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()}
            movable_work = movable.repeat_interleave(P, dim=0)
        else:
            work = batch
            movable_work = movable

        def fwd(x_t, t, _b=work, _m=movable_work):
            extra = {}
            if use_ads_ref:
                extra["ads_ref_pos"] = _b["ads_ref_pos"]
            return model(
                pos=_b["pos"],
                x_t=x_t,
                t=t,
                atomic_numbers=_b["atomic_numbers"],
                tags=_b["tags"],
                movable_mask=_m,
                pad_mask=_b["pad_mask"],
                cell=_b["cell"],
                **extra,
            )

        # FK steering config (energy_fn captures replicated context).
        fk_cfg = None
        if args.fk_particles > 0:
            from adsorbgen.evaluation.energy import make_fk_energy_fn
            energy_fn = make_fk_energy_fn(
                fk_energy_model, work["atomic_numbers"], work["cell"],
            )
            fk_cfg = FKSteeringConfig(
                num_particles=args.fk_particles,
                energy_fn=energy_fn,
                fk_lambda=args.fk_lambda,
                resampling_interval=args.fk_resample_interval,
                fk_start_time=args.fk_start_time,
                potential_mode=args.fk_potential,
            )

        x_out = euler_sample(
            fwd,
            work["pos"],
            movable_work,
            work["pad_mask"],
            flow_cfg,
            num_steps=args.flow_steps,
            use_sde=args.use_sde,
            refine_final=args.refine_final,
            sde_schedule=args.sde_schedule,
            sde_alpha=args.sde_alpha,
            fk_steering=fk_cfg,
        )

        # Fix A: argmin-energy particle selection (AtomMOF-style) instead of
        # keeping particle 0. Computes final UMA energy on all P particles
        # then picks the best per (BK,) group.
        if P > 1 and fk_energy_model is not None:
            with torch.no_grad():
                e_final = fk_energy_model(
                    x_out, work["cell"], work["atomic_numbers"], work["pad_mask"],
                )
            e_final = e_final.view(BK, P)
            best_p = torch.argmin(e_final, dim=1)
            row_idx = torch.arange(BK, device=x_out.device) * P + best_p
            x_out = x_out.index_select(0, row_idx)

        relax_jobs = []
        for i, (global_i, sys_i, sample_i, rep, source_entry) in enumerate(metas):
            n = int(batch["pad_mask"][i].sum().item())
            tags = batch["tags"][i, :n].detach().cpu().numpy().astype(np.int64)
            numbers = batch["atomic_numbers"][i, :n].detach().cpu().numpy().astype(np.int64)
            fixed = batch["fixed"][i, :n].detach().cpu().numpy().astype(np.int64)
            cell = batch["cell"][i].detach().cpu().numpy()
            if cell.ndim == 3:
                cell = cell[0]
            pos_ref = batch["pos"][i, :n].detach().cpu().numpy().astype(np.float64)
            pos_pred = x_out[i, :n].detach().cpu().numpy().astype(np.float64)
            pos_gt = batch["pos_relaxed"][i, :n].detach().cpu().numpy().astype(np.float64)
            ads_id = int(batch["ads_id"][i].item()) if "ads_id" in batch else int(samples[i]["ads_id"].item())
            relax_jobs.append({
                "meta": (global_i, sys_i, sample_i, rep, source_entry),
                "numbers": numbers,
                "tags": tags,
                "fixed": fixed,
                "cell": cell,
                "pos_ref": pos_ref,
                "pos_gt": pos_gt,
                "ads_id": ads_id,
                "relax_input": {
                    "numbers": numbers,
                    "tags": tags,
                    "fixed": fixed,
                    "cell": cell,
                    "pos_pred": pos_pred,
                },
            })

        relaxed_results = relax_lbfgs_batch(
            relax_jobs,
            calc,
            args,
            batcher=batcher,
            calculator_cls=FAIRChemCalculator,
        )

        for job, relaxed in zip(relax_jobs, relaxed_results, strict=True):
            global_i, sys_i, sample_i, rep, source_entry = job["meta"]
            numbers = job["numbers"]
            tags = job["tags"]
            fixed = job["fixed"]
            cell = job["cell"]
            pos_ref = job["pos_ref"]
            pos_gt = job["pos_gt"]
            ads_id = job["ads_id"]

            status = "ok"
            valid = False
            success = False
            anomaly = None
            e_ref = float(rep["E_sys_ref"])
            e_sys = float(relaxed["E_sys"])
            improvement = float(e_ref - e_sys) if np.isfinite(e_sys) else float("nan")
            if not relaxed["converged"] or not np.isfinite(e_sys):
                status = "uma_unconverged"
            else:
                ar = _score_record_anomaly({
                    "sid": int(rep["sid"]),
                    "system_key": tuple(rep["system_key"]),
                    "ads_id": ads_id,
                    "pos_ref": torch.as_tensor(pos_ref, dtype=torch.float32),
                    "pos_pred": torch.as_tensor(relaxed["pos_relaxed"], dtype=torch.float32),
                    "pos_gt": torch.as_tensor(pos_gt, dtype=torch.float32),
                    "atomic_numbers": torch.as_tensor(numbers, dtype=torch.long),
                    "tags": torch.as_tensor(tags, dtype=torch.long),
                    "cell": torch.as_tensor(cell, dtype=torch.float32),
                })
                valid, anomaly = status_from_anomaly(ar)
                if valid and improvement > args.success_margin:
                    success = True
                elif not valid:
                    status = str(anomaly)

            row = {
                "global_i": int(global_i),
                "system_i": int(sys_i),
                "sample_i": int(sample_i),
                "system_key": list(rep["system_key"]),
                "lmdb_id": int(rep["lmdb_id"]),
                "raw_idx": int(rep["raw_idx"]),
                "sid": int(rep["sid"]),
                "ads_id": int(ads_id),
                "E_sys": e_sys,
                "E_sys_ref": e_ref,
                "improvement": improvement,
                "fmax": float(relaxed["fmax"]),
                "n_steps": int(relaxed["n_steps"]),
                "converged": bool(relaxed["converged"]),
                "valid": bool(valid),
                "success": bool(success),
                "status": status,
                "anomaly": anomaly,
                "relax_error": relaxed["error"],
            }
            rows.append(row)
            if success:
                payload = {
                    **row,
                    "pos_relaxed": np.asarray(relaxed["pos_relaxed"], dtype=np.float32),
                    "tags": tags.astype(np.int64),
                    "atomic_numbers": numbers.astype(np.int64),
                    "fixed": fixed.astype(np.int64),
                    "cell": np.asarray(cell, dtype=np.float32),
                    "source_pos": np.asarray(source_entry["pos"], dtype=np.float32),
                    "source_pos_relaxed": np.asarray(source_entry["pos_relaxed"], dtype=np.float32),
                }
                success_entries.append(payload)
            if (
                args.save_window_candidates
                and bool(valid)
                and bool(relaxed["converged"])
                and np.isfinite(e_sys)
                and e_sys <= e_ref + float(args.candidate_window_ev)
            ):
                window_candidate_entries.append({
                    **row,
                    "pos_relaxed": np.asarray(relaxed["pos_relaxed"], dtype=np.float32),
                    "tags": tags.astype(np.int64),
                    "atomic_numbers": numbers.astype(np.int64),
                    "fixed": fixed.astype(np.int64),
                    "cell": np.asarray(cell, dtype=np.float32),
                    "source_pos": np.asarray(source_entry["pos"], dtype=np.float32),
                    "source_pos_relaxed": np.asarray(source_entry["pos_relaxed"], dtype=np.float32),
                    "candidate_window_ev": float(args.candidate_window_ev),
                })

        if len(rows) % max(1, args.progress_every) == 0 or start + args.flow_batch_size >= len(tasks):
            write_progress(progress_path, args, rows, success_entries, selected_systems, t0)
            with (out_dir / f"success_shard{args.shard_idx}.pkl").open("wb") as f:
                pickle.dump(success_entries, f, protocol=pickle.HIGHEST_PROTOCOL)
            if args.save_window_candidates:
                with (out_dir / f"candidate_shard{args.shard_idx}.pkl").open("wb") as f:
                    pickle.dump(window_candidate_entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    for env in source_envs:
        env.close()

    summary = summarize_rows(rows, selected_systems)
    summary.update({
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
        "elapsed_sec": time.time() - t0,
        "ckpt": str(args.ckpt),
        "lbfgs": {
            "fmax": args.uma_fmax,
            "max_steps": args.uma_max_steps,
            "maxstep": args.lbfgs_maxstep,
            "memory": args.lbfgs_memory,
            "damping": args.lbfgs_damping,
            "alpha": args.lbfgs_alpha,
            "use_fairchem_batcher": bool(args.use_fairchem_batcher),
            "concurrency": int(args.lbfgs_concurrency),
            "batcher_max_atoms": int(args.batcher_max_atoms),
            "batcher_wait_timeout_s": float(args.batcher_wait_timeout_s),
        },
        "sampling": {
            "seed": args.seed,
            "flow_steps": args.flow_steps,
            "flow_batch_size": args.flow_batch_size,
            "prior_mode": args.prior_mode,
            "use_sde": args.use_sde,
            "refine_final": args.refine_final,
            "sde_schedule": args.sde_schedule,
            "sde_alpha": args.sde_alpha,
            "fk_particles": args.fk_particles,
            "fk_lambda": args.fk_lambda,
            "fk_resample_interval": args.fk_resample_interval,
            "fk_start_time": args.fk_start_time,
            "fk_potential": args.fk_potential,
            "fk_energy": args.fk_energy,
            "fk_uma_model": args.fk_uma_model,
        },
        "moving_window_candidates": {
            "enabled": bool(args.save_window_candidates),
            "candidate_window_ev": float(args.candidate_window_ev),
            "saved_entries": len(window_candidate_entries),
        },
    })

    with (out_dir / f"rows_shard{args.shard_idx}.pkl").open("wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (out_dir / f"success_shard{args.shard_idx}.pkl").open("wb") as f:
        pickle.dump(success_entries, f, protocol=pickle.HIGHEST_PROTOCOL)
    if args.save_window_candidates:
        with (out_dir / f"candidate_shard{args.shard_idx}.pkl").open("wb") as f:
            pickle.dump(window_candidate_entries, f, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_write_json(out_dir / f"shard_{args.shard_idx}.json", summary)
    write_progress(progress_path, args, rows, success_entries, selected_systems, t0)
    if batcher is not None:
        batcher.shutdown(wait=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
