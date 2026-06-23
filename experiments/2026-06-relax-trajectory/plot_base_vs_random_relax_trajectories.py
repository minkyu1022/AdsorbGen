#!/usr/bin/env python
"""Compare relaxation trajectories from base-model and random-placement starts.

For N randomly selected systems and K deterministic placements per system:
  1. save the exact placement batch used for both methods,
  2. generate a base model x1 prediction from that placement,
  3. relax both the model output and the raw random placement with UMA ASE LBFGS,
  4. plot mean energy and mean RMSD-to-own-final trajectory per system.

RMSD is computed against each candidate's own converged/final configuration,
then averaged over placements. This measures how quickly each start approaches
its own local minimum, not whether both methods reach the same minimum.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from tqdm.auto import tqdm


REPO = Path(os.environ.get("ADSGEN_ROOT", "/home1/irteam/AdsorbGen")).resolve()
GEOOPT = REPO / "geoopt"
for path in (REPO, GEOOPT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement  # noqa: E402
from adsorbgen.evaluation.energy import UMAForce  # noqa: E402
from adsorbgen.flow import euler_sample  # noqa: E402
from adsorbgen.replay.eval import _model_cfg, _runtime_movable_mask  # noqa: E402
from geoopt import load_model_from_ckpt  # noqa: E402


def read_lmdb_entry(env: lmdb.Environment, idx: int) -> dict:
    with env.begin() as txn:
        raw = txn.get(str(int(idx)).encode("ascii"))
    if raw is None:
        raise KeyError(f"missing LMDB key {idx}")
    return pickle.loads(raw)


def lmdb_length(env: lmdb.Environment) -> int:
    with env.begin() as txn:
        raw = txn.get(b"length")
    if raw is None:
        raise KeyError("LMDB is missing key b'length'")
    return int(pickle.loads(raw))


def system_key_of(entry: dict, idx: int) -> str:
    return str(entry.get("system_key", entry.get("sid", idx)))


def select_unique_systems(lmdb_path: str, n_systems: int, seed: int) -> list[dict[str, Any]]:
    """Randomly select unique system_key rows from an LMDB."""
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)
    n = lmdb_length(env)
    first_by_system: dict[str, int] = {}
    for idx in range(n):
        entry = read_lmdb_entry(env, idx)
        key = system_key_of(entry, idx)
        if key not in first_by_system:
            first_by_system[key] = idx
    env.close()
    keys = np.array(sorted(first_by_system), dtype=object)
    if len(keys) < n_systems:
        raise ValueError(f"only {len(keys)} unique systems in {lmdb_path}, need {n_systems}")
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(keys, size=n_systems, replace=False).tolist())
    return [{"system_key": str(k), "raw_idx": int(first_by_system[str(k)])} for k in chosen]


def load_or_create_selected(args, out_dir: Path) -> list[dict[str, Any]]:
    if args.selected_systems_json:
        payload = json.loads(Path(args.selected_systems_json).read_text())
        return list(payload["systems"])
    selected_path = out_dir / "selected_systems.json"
    if selected_path.exists():
        payload = json.loads(selected_path.read_text())
        systems = list(payload["systems"])
        if len(systems) >= int(args.num_systems):
            return systems[: int(args.num_systems)]
    selected = select_unique_systems(args.lmdb, int(args.num_systems), int(args.seed))
    selected_path.write_text(
        json.dumps({"seed": args.seed, "systems": selected}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return selected


def cell3(cell: np.ndarray) -> np.ndarray:
    arr = np.asarray(cell, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def atoms_from_arrays(numbers, positions, cell, tags, fixed) -> Atoms:
    atoms = Atoms(
        numbers=np.asarray(numbers, dtype=int),
        positions=np.asarray(positions, dtype=np.float64),
        cell=np.asarray(cell, dtype=np.float64),
        pbc=True,
        tags=np.asarray(tags, dtype=int).tolist(),
    )
    fixed = np.asarray(fixed, dtype=bool)
    if not fixed.any():
        fixed = np.asarray(tags, dtype=int) == 0
    if fixed.any():
        atoms.set_constraint(FixAtoms(indices=np.where(fixed)[0].tolist()))
    return atoms


def mic_delta(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(np.asarray(cell, dtype=np.float64))
    frac = delta @ inv
    frac = frac - np.round(frac)
    return frac @ np.asarray(cell, dtype=np.float64)


def rmsd_to_final(
    positions: list[np.ndarray],
    final_pos: np.ndarray,
    cell: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    out = []
    mask = np.asarray(mask, dtype=bool)
    denom = max(int(mask.sum()), 1)
    for pos in positions:
        delta = mic_delta(np.asarray(pos, dtype=np.float64) - final_pos, cell)
        out.append(float(np.sqrt((delta[mask] ** 2).sum() / denom)))
    return np.asarray(out, dtype=np.float64)


def interpolate_nan_curves(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_len = max((len(c) for c in curves), default=0)
    if max_len == 0:
        return np.arange(0), np.array([]), np.array([])
    mat = np.full((len(curves), max_len), np.nan, dtype=np.float64)
    for i, curve in enumerate(curves):
        mat[i, : len(curve)] = curve
        if len(curve) < max_len and len(curve) > 0:
            mat[i, len(curve):] = curve[-1]
    x = np.arange(max_len)
    mean = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(np.isfinite(mat).sum(axis=0), 1))
    return x, mean, sem


def relax_with_trajectory(atoms: Atoms, calc, args) -> dict[str, Any]:
    atoms = atoms.copy()
    atoms.calc = calc
    energies: list[float] = []
    positions: list[np.ndarray] = []

    def record() -> None:
        try:
            energies.append(float(atoms.get_potential_energy()))
            positions.append(atoms.get_positions().astype(np.float64, copy=True))
        except Exception:
            energies.append(float("nan"))
            positions.append(atoms.get_positions().astype(np.float64, copy=True))

    record()
    opt = LBFGS(
        atoms,
        logfile=None,
        maxstep=float(args.lbfgs_maxstep),
        memory=int(args.lbfgs_memory),
        damping=float(args.lbfgs_damping),
        alpha=float(args.lbfgs_alpha),
    )
    opt.attach(record, interval=1)
    err = None
    try:
        converged = bool(opt.run(fmax=float(args.fmax), steps=int(args.max_steps)))
        forces = atoms.get_forces()
        fmax = float(np.linalg.norm(forces, axis=1).max())
    except Exception as exc:
        converged = False
        fmax = float("nan")
        err = repr(exc)
    final_pos = atoms.get_positions().astype(np.float64, copy=True)
    final_energy = float(energies[-1]) if energies else float("nan")
    return {
        "energies": np.asarray(energies, dtype=np.float64),
        "positions": positions,
        "final_pos": final_pos,
        "final_energy": final_energy,
        "converged": bool(converged),
        "n_steps": int(getattr(opt, "nsteps", 0)),
        "fmax": fmax,
        "error": err,
    }


@torch.no_grad()
def build_base_predictions(args, device, tasks, placement_ds, model, flow_cfg):
    model_cfg = _model_cfg(model)
    use_ads_ref = bool(getattr(model_cfg, "use_ads_ref_pos", False))
    langevin_force_model = None
    if bool(getattr(model_cfg, "use_langevin_param", False)):
        if str(getattr(model_cfg, "langevin_eval_on", "x_t")) != "x_t":
            raise ValueError("Only langevin_eval_on='x_t' is implemented")
        langevin_force_model = UMAForce(
            model_name=str(args.langevin_uma_model),
            task_name=str(args.langevin_uma_task),
            device=str(device),
        )

    jobs = []
    for start in tqdm(range(0, len(tasks), int(args.flow_batch_size)), desc="base inference"):
        chunk = tasks[start:start + int(args.flow_batch_size)]
        samples = []
        metas = []
        for task in chunk:
            seed_i = (int(args.seed) + int(task["global_i"])) & 0xFFFF_FFFF
            np.random.seed(seed_i)
            random.seed(seed_i)
            samples.append(placement_ds[int(task["raw_idx"])])
            metas.append(task)

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
            out = model(
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
            return out["pred_x1"] if isinstance(out, dict) else out

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

        for i, task in enumerate(metas):
            n = int(batch["pad_mask"][i].sum().item())
            jobs.append({
                **task,
                "numbers": batch["atomic_numbers"][i, :n].detach().cpu().numpy().astype(np.int64),
                "tags": batch["tags"][i, :n].detach().cpu().numpy().astype(np.int64),
                "fixed": batch["fixed"][i, :n].detach().cpu().numpy().astype(np.int64),
                "cell": cell3(batch["cell"][i].detach().cpu().numpy()),
                "pos_prior": batch["pos"][i, :n].detach().cpu().numpy().astype(np.float64),
                "pos_base": x_out[i, :n].detach().cpu().numpy().astype(np.float64),
                "pos_lmdb_relaxed": batch["pos_relaxed"][i, :n].detach().cpu().numpy().astype(np.float64),
            })
    return jobs


def plot_system_curves(out_dir: Path, system_key: str, method_payload: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(system_key))[:160]
    sys_dir = out_dir / "plots" / safe
    sys_dir.mkdir(parents=True, exist_ok=True)

    def label(method: str) -> str:
        return "ours" if method == "base" else method

    for method, records in method_payload.items():
        e_curves = [np.asarray(r["energies"], dtype=np.float64) for r in records]
        r_curves = [np.asarray(r["rmsd_to_final"], dtype=np.float64) for r in records]
        for name, curves, ylabel in [
            ("energy", e_curves, "UMA E_sys (eV)"),
            ("rmsd_to_own_final", r_curves, "RMSD to own final (A)"),
        ]:
            x, mean, sem = interpolate_nan_curves(curves)
            fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=160)
            ax.plot(x, mean, lw=2.0)
            if len(x):
                ax.fill_between(x, mean - sem, mean + sem, alpha=0.2)
            ax.set_xlabel("LBFGS step")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{label(method)} | {system_key} | mean over placements")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(sys_dir / f"{label(method)}_{name}.png")
            plt.close(fig)

    for name, key, ylabel, filename in [
        ("energy", "energies", "UMA E_sys (eV)", "energy_ours_vs_random.png"),
        ("rmsd_to_own_final", "rmsd_to_final", "RMSD to own final (A)", "rmsd_ours_vs_random.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=160)
        for method, records in method_payload.items():
            x, mean, sem = interpolate_nan_curves([np.asarray(r[key]) for r in records])
            ax.plot(x, mean, lw=2.0, label=label(method))
            if len(x):
                ax.fill_between(x, mean - sem, mean + sem, alpha=0.15)
        ax.set_xlabel("LBFGS step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{system_key} | {name} | mean over placements")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(sys_dir / filename)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=160)
    for method, records in method_payload.items():
        x, mean, sem = interpolate_nan_curves([np.asarray(r["energies"]) for r in records])
        axes[0].plot(x, mean, lw=2.0, label=label(method))
        if len(x):
            axes[0].fill_between(x, mean - sem, mean + sem, alpha=0.15)
        x, mean, sem = interpolate_nan_curves([np.asarray(r["rmsd_to_final"]) for r in records])
        axes[1].plot(x, mean, lw=2.0, label=label(method))
        if len(x):
            axes[1].fill_between(x, mean - sem, mean + sem, alpha=0.15)
    axes[0].set_xlabel("LBFGS step")
    axes[0].set_ylabel("UMA E_sys (eV)")
    axes[1].set_xlabel("LBFGS step")
    axes[1].set_ylabel("RMSD to own final (A)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"{system_key} | mean over placements")
    fig.tight_layout()
    fig.savefig(sys_dir / "combined_base_vs_random.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/irteam/runs/training/base/base.ckpt")
    ap.add_argument("--lmdb", default="/home1/irteam/data/processed_old/oc20dense.lmdb")
    ap.add_argument("--out-dir", default="/home1/irteam/data/replay/base_vs_random_relax_traj_5x5")
    ap.add_argument("--num-systems", type=int, default=5)
    ap.add_argument("--num-placements", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--flow-batch-size", type=int, default=8)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--langevin-uma-model", default="uma-s-1p2")
    ap.add_argument("--langevin-uma-task", default="oc20")
    ap.add_argument("--lbfgs-maxstep", type=float, default=0.2)
    ap.add_argument("--lbfgs-memory", type=int, default=100)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    ap.add_argument("--rmsd-scope", choices=["movable", "adsorbate", "all"], default="movable")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge-shards", action="store_true")
    ap.add_argument("--selected-systems-json", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_shards:
        selected_path = out_dir / "selected_systems.json"
        if not selected_path.exists():
            raise FileNotFoundError(selected_path)
        shard_paths = sorted((out_dir / "shards").glob("shard_*.json"))
        if not shard_paths:
            raise FileNotFoundError(f"no shard_*.json under {out_dir / 'shards'}")
        results = []
        for path in shard_paths:
            results.extend(json.loads(path.read_text()))
        by_system_method: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for rec in results:
            by_system_method[str(rec["system_key"])][str(rec["method"])].append(rec)
        summaries = {}
        for system_key, payload in by_system_method.items():
            plot_system_curves(out_dir, system_key, payload)
            summaries[system_key] = {}
            for method, recs in payload.items():
                final_e = np.asarray([r["final_energy"] for r in recs], dtype=np.float64)
                summaries[system_key][method] = {
                    "n": len(recs),
                    "n_converged": int(sum(bool(r["converged"]) for r in recs)),
                    "final_energy_mean_eV": float(np.nanmean(final_e)),
                    "final_energy_std_eV": float(np.nanstd(final_e)),
                    "n_steps_mean": float(np.mean([r["n_steps"] for r in recs])),
                }
        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps({"args": vars(args), "systems": summaries}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps({"out_dir": str(out_dir), "systems": summaries}, indent=2, sort_keys=True))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for UMA/model inference")

    selected = load_or_create_selected(args, out_dir)
    tasks = []
    for sys_i, rec in enumerate(selected):
        for sample_i in range(int(args.num_placements)):
            task = {
                "global_i": sys_i * int(args.num_placements) + sample_i,
                "system_i": sys_i,
                "system_key": rec["system_key"],
                "raw_idx": rec["raw_idx"],
                "sample_i": sample_i,
            }
            if int(task["global_i"]) % int(args.num_shards) == int(args.shard_idx):
                tasks.append(task)
    print(
        f"[shard {args.shard_idx}/{args.num_shards}] tasks={len(tasks)}/"
        f"{len(selected) * int(args.num_placements)}",
        flush=True,
    )

    model, flow_cfg = load_model_from_ckpt(Path(args.ckpt), device)
    model_cfg = _model_cfg(model)
    placement_ds = PlacementPriorDataset(
        args.lmdb,
        prior_mode=str(args.prior_mode),
        max_samples=None,
        provide_ads_ref_pos=bool(getattr(model_cfg, "use_ads_ref_pos", False)),
        skip_anomaly=False,
        slab_source=str(getattr(model, "adsorbgen_slab_source", "initial")),
        pristine_slabs=str(getattr(model, "adsorbgen_pristine_slabs", "")),
        pristine_index=str(getattr(model, "adsorbgen_pristine_index", "")),
    )
    jobs = build_base_predictions(args, device, tasks, placement_ds, model, flow_cfg)

    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    predict_unit = pretrained_mlip.get_predict_unit(str(args.uma_model), device=str(device))
    calc = FAIRChemCalculator(predict_unit, task_name=str(args.uma_task))

    results: list[dict[str, Any]] = []
    by_system_method: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for job in tqdm(jobs, desc="UMA ASE-LBFGS trajectories"):
        for method, start_key in [("base", "pos_base"), ("random", "pos_prior")]:
            atoms = atoms_from_arrays(
                job["numbers"], job[start_key], job["cell"], job["tags"], job["fixed"],
            )
            traj = relax_with_trajectory(atoms, calc, args)
            if args.rmsd_scope == "all":
                mask = np.ones_like(job["tags"], dtype=bool)
            elif args.rmsd_scope == "adsorbate":
                mask = np.asarray(job["tags"]) == 2
            else:
                mask = np.asarray(job["fixed"]) == 0
                if not mask.any():
                    mask = np.asarray(job["tags"]) >= 1
            rmsd = rmsd_to_final(traj["positions"], traj["final_pos"], job["cell"], mask)
            rec = {
                "method": method,
                "system_key": job["system_key"],
                "system_i": int(job["system_i"]),
                "sample_i": int(job["sample_i"]),
                "raw_idx": int(job["raw_idx"]),
                "converged": bool(traj["converged"]),
                "n_steps": int(traj["n_steps"]),
                "fmax": float(traj["fmax"]),
                "final_energy": float(traj["final_energy"]),
                "error": traj["error"],
                "energies": traj["energies"].tolist(),
                "rmsd_to_final": rmsd.tolist(),
            }
            results.append(rec)
            by_system_method[job["system_key"]][method].append(rec)

            cand_dir = out_dir / "candidates" / f"system{job['system_i']:02d}_sample{job['sample_i']:02d}_{method}"
            cand_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cand_dir / "trajectory.npz",
                energies=traj["energies"],
                rmsd_to_final=rmsd,
                final_pos=traj["final_pos"],
                start_pos=job[start_key],
                prior_pos=job["pos_prior"],
                base_pos=job["pos_base"],
                lmdb_relaxed_pos=job["pos_lmdb_relaxed"],
                numbers=job["numbers"],
                tags=job["tags"],
                fixed=job["fixed"],
                cell=job["cell"],
            )

    selected_payload = {"seed": args.seed, "systems": selected}
    (out_dir / "selected_systems.json").write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / f"shard_{int(args.shard_idx):03d}.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    if int(args.num_shards) > 1:
        print(json.dumps({
            "out_dir": str(out_dir),
            "shard_idx": int(args.shard_idx),
            "n_records": len(results),
            "merge_command": f"{sys.executable} {Path(__file__).resolve()} --out-dir {out_dir} --merge-shards",
        }, indent=2, sort_keys=True))
        return

    summaries = {}
    for system_key, payload in by_system_method.items():
        plot_system_curves(out_dir, system_key, payload)
        summaries[system_key] = {}
        for method, recs in payload.items():
            final_e = np.asarray([r["final_energy"] for r in recs], dtype=np.float64)
            summaries[system_key][method] = {
                "n": len(recs),
                "n_converged": int(sum(bool(r["converged"]) for r in recs)),
                "final_energy_mean_eV": float(np.nanmean(final_e)),
                "final_energy_std_eV": float(np.nanstd(final_e)),
                "n_steps_mean": float(np.mean([r["n_steps"] for r in recs])),
                "plot_dir": str(out_dir / "plots" / "".join(c if c.isalnum() or c in "-_." else "_" for c in str(system_key))[:160]),
            }
    (out_dir / "selected_systems.json").write_text(
        json.dumps({"seed": args.seed, "systems": selected}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps({"args": vars(args), "systems": summaries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "systems": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
