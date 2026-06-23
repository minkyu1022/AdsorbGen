#!/usr/bin/env python
"""Compare uninterrupted LBFGS trajectories with midpoint-restarted LBFGS."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.optimize import LBFGS
from tqdm.auto import tqdm


BASE_SCRIPT = Path(__file__).with_name("plot_base_vs_random_relax_trajectories.py")
spec = importlib.util.spec_from_file_location("relax_traj_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to import {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
sys.modules["relax_traj_base"] = base
spec.loader.exec_module(base)


def label(method: str) -> str:
    return "ours" if method == "base" else method


def load_full_records(out_dir: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    path = out_dir / "results.json"
    if not path.exists():
        raise FileNotFoundError(path)
    records = json.loads(path.read_text())
    out = {}
    for rec in records:
        key = (int(rec["system_i"]), int(rec["sample_i"]), str(rec["method"]))
        out[key] = rec
    return out


def run_fixed_steps(atoms, calc, args, steps: int) -> dict[str, Any]:
    atoms = atoms.copy()
    atoms.calc = calc
    energies: list[float] = []
    positions: list[np.ndarray] = []

    def record() -> None:
        try:
            energies.append(float(atoms.get_potential_energy()))
        except Exception:
            energies.append(float("nan"))
        positions.append(atoms.get_positions().astype(np.float64, copy=True))

    record()
    if steps <= 0:
        return {"energies": np.asarray(energies), "positions": positions, "atoms": atoms}
    opt = LBFGS(
        atoms,
        logfile=None,
        maxstep=float(args.lbfgs_maxstep),
        memory=int(args.lbfgs_memory),
        damping=float(args.lbfgs_damping),
        alpha=float(args.lbfgs_alpha),
    )
    opt.attach(record, interval=1)
    opt.run(fmax=0.0, steps=int(steps))
    return {"energies": np.asarray(energies), "positions": positions, "atoms": atoms}


def merge_curves(first: dict[str, Any], second: dict[str, Any], final_pos, cell, mask):
    energies = np.concatenate([
        np.asarray(first["energies"], dtype=np.float64),
        np.asarray(second["energies"], dtype=np.float64)[1:],
    ])
    positions = list(first["positions"]) + list(second["positions"])[1:]
    rmsd = base.rmsd_to_final(positions, final_pos, cell, mask)
    return energies, rmsd


def plot_restart_curves(out_dir: Path, system_key: str, records: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(system_key))[:160]
    sys_dir = out_dir / "restart_plots" / safe
    sys_dir.mkdir(parents=True, exist_ok=True)

    styles = {
        ("base", "full"): ("tab:blue", "-", "ours full"),
        ("base", "restart_mid"): ("tab:blue", "--", "ours restart@mid"),
        ("random", "full"): ("tab:orange", "-", "random full"),
        ("random", "restart_mid"): ("tab:orange", "--", "random restart@mid"),
    }
    for key, ylabel, filename in [
        ("energies", "UMA E_sys (eV)", "energy_full_vs_midrestart.png"),
        ("rmsd_to_full_final", "RMSD to uninterrupted final (A)", "rmsd_full_vs_midrestart.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4.3), dpi=160)
        for method in ["base", "random"]:
            for variant in ["full", "restart_mid"]:
                recs = records.get(f"{method}:{variant}", [])
                if not recs:
                    continue
                x, mean, sem = base.interpolate_nan_curves([
                    np.asarray(r[key], dtype=np.float64) for r in recs
                ])
                color, ls, name = styles[(method, variant)]
                ax.plot(x, mean, lw=2.0, ls=ls, color=color, label=name)
                if len(x):
                    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.12)
        ax.set_xlabel("cumulative LBFGS step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{system_key} | full vs midpoint restart | mean over placements")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(sys_dir / filename)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), dpi=160)
    for ax, key, ylabel, title in [
        (axes[0], "energies", "UMA E_sys (eV)", "Energy"),
        (axes[1], "rmsd_to_full_final", "RMSD to uninterrupted final (A)", "RMSD"),
    ]:
        for method in ["base", "random"]:
            for variant in ["full", "restart_mid"]:
                recs = records.get(f"{method}:{variant}", [])
                if not recs:
                    continue
                x, mean, sem = base.interpolate_nan_curves([
                    np.asarray(r[key], dtype=np.float64) for r in recs
                ])
                color, ls, name = styles[(method, variant)]
                ax.plot(x, mean, lw=2.0, ls=ls, color=color, label=name)
                if len(x):
                    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.12)
        ax.set_xlabel("cumulative LBFGS step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"{system_key} | full vs midpoint restart | mean over placements")
    fig.tight_layout()
    fig.savefig(sys_dir / "combined_full_vs_midrestart.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-out-dir", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--ckpt", default="/home/irteam/runs/training/base/base.ckpt")
    ap.add_argument("--lmdb", default="/home1/irteam/data/processed_old/oc20dense.lmdb")
    ap.add_argument("--selected-systems-json", default="")
    ap.add_argument("--num-systems", type=int, default=5)
    ap.add_argument("--num-placements", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--flow-batch-size", type=int, default=4)
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
    args = ap.parse_args()

    source_out = Path(args.source_out_dir)
    out_dir = Path(args.out_dir or (str(source_out).rstrip("/") + "_midrestart"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_shards:
        shard_paths = sorted((out_dir / "shards").glob("shard_*.json"))
        if not shard_paths:
            raise FileNotFoundError(f"no shard_*.json under {out_dir / 'shards'}")
        results = []
        for path in shard_paths:
            results.extend(json.loads(path.read_text()))
        grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for rec in results:
            grouped[str(rec["system_key"])][f"{rec['method']}:{rec['variant']}"].append(rec)
        summaries = {}
        for system_key, payload in grouped.items():
            plot_restart_curves(out_dir, system_key, payload)
            summaries[system_key] = {}
            for group_key, recs in payload.items():
                summaries[system_key][group_key] = {
                    "n": len(recs),
                    "n_converged": int(sum(bool(r.get("converged", False)) for r in recs)),
                    "n_steps_mean": float(np.mean([r["n_steps"] for r in recs])),
                    "final_energy_mean_eV": float(np.nanmean([r["final_energy"] for r in recs])),
                }
        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps({"source_out_dir": str(source_out), "systems": summaries}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps({"out_dir": str(out_dir), "systems": summaries}, indent=2, sort_keys=True))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")

    if not args.selected_systems_json:
        args.selected_systems_json = str(source_out / "selected_systems.json")
    selected = base.load_or_create_selected(args, out_dir)
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
    print(f"[shard {args.shard_idx}/{args.num_shards}] tasks={len(tasks)}", flush=True)

    full = load_full_records(source_out)
    model, flow_cfg = base.load_model_from_ckpt(Path(args.ckpt), device)
    model_cfg = base._model_cfg(model)
    placement_ds = base.PlacementPriorDataset(
        args.lmdb,
        prior_mode=str(args.prior_mode),
        max_samples=None,
        provide_ads_ref_pos=bool(getattr(model_cfg, "use_ads_ref_pos", False)),
        skip_anomaly=False,
        slab_source=str(getattr(model, "adsorbgen_slab_source", "initial")),
        pristine_slabs=str(getattr(model, "adsorbgen_pristine_slabs", "")),
        pristine_index=str(getattr(model, "adsorbgen_pristine_index", "")),
    )
    jobs = base.build_base_predictions(args, device, tasks, placement_ds, model, flow_cfg)

    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    predict_unit = pretrained_mlip.get_predict_unit(str(args.uma_model), device=str(device))
    calc = FAIRChemCalculator(predict_unit, task_name=str(args.uma_task))

    results = []
    for job in tqdm(jobs, desc="midpoint restart"):
        for method, start_key in [("base", "pos_base"), ("random", "pos_prior")]:
            fkey = (int(job["system_i"]), int(job["sample_i"]), method)
            full_rec = full[fkey]
            cand_path = (
                source_out / "candidates" /
                f"system{job['system_i']:02d}_sample{job['sample_i']:02d}_{method}" /
                "trajectory.npz"
            )
            cand = np.load(cand_path)
            final_pos = np.asarray(cand["final_pos"], dtype=np.float64)
            if args.rmsd_scope == "all":
                mask = np.ones_like(job["tags"], dtype=bool)
            elif args.rmsd_scope == "adsorbate":
                mask = np.asarray(job["tags"]) == 2
            else:
                mask = np.asarray(job["fixed"]) == 0
                if not mask.any():
                    mask = np.asarray(job["tags"]) >= 1

            full_energies = np.asarray(full_rec["energies"], dtype=np.float64)
            full_rmsd = np.asarray(full_rec["rmsd_to_final"], dtype=np.float64)
            results.append({
                "variant": "full",
                "method": method,
                "system_key": job["system_key"],
                "system_i": int(job["system_i"]),
                "sample_i": int(job["sample_i"]),
                "n_steps": int(full_rec["n_steps"]),
                "mid_step": int(full_rec["n_steps"]) // 2,
                "converged": bool(full_rec["converged"]),
                "final_energy": float(full_rec["final_energy"]),
                "energies": full_energies.tolist(),
                "rmsd_to_full_final": full_rmsd.tolist(),
            })

            mid_step = max(int(full_rec["n_steps"]) // 2, 0)
            atoms0 = base.atoms_from_arrays(
                job["numbers"], job[start_key], job["cell"], job["tags"], job["fixed"],
            )
            first = run_fixed_steps(atoms0, calc, args, mid_step)
            second = base.relax_with_trajectory(first["atoms"], calc, args)
            restart_energies, restart_rmsd = merge_curves(first, second, final_pos, job["cell"], mask)
            total_steps = mid_step + int(second["n_steps"])
            results.append({
                "variant": "restart_mid",
                "method": method,
                "system_key": job["system_key"],
                "system_i": int(job["system_i"]),
                "sample_i": int(job["sample_i"]),
                "n_steps": int(total_steps),
                "mid_step": int(mid_step),
                "restart_steps": int(second["n_steps"]),
                "converged": bool(second["converged"]),
                "final_energy": float(second["final_energy"]),
                "full_final_energy": float(full_rec["final_energy"]),
                "delta_final_energy_vs_full_eV": float(second["final_energy"] - full_rec["final_energy"]),
                "energies": restart_energies.tolist(),
                "rmsd_to_full_final": restart_rmsd.tolist(),
            })

    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / f"shard_{int(args.shard_idx):03d}.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "shard_idx": int(args.shard_idx), "n_records": len(results)}, indent=2))


if __name__ == "__main__":
    main()
