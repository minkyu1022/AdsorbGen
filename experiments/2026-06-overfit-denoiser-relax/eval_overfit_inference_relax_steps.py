#!/usr/bin/env python
"""Run overfit checkpoint inference and measure post-relaxation LBFGS steps."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from torch.utils.data import DataLoader


def install_imports(repo: Path, adsorbates_pkl: Path) -> None:
    os.environ["ADSGEN_ROOT"] = str(repo)
    os.environ["ADSORBATES_PKL"] = str(adsorbates_pkl)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def cell3(cell: np.ndarray) -> np.ndarray:
    arr = np.asarray(cell, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def atoms_from_arrays(numbers, pos, cell, tags, fixed) -> Atoms:
    atoms = Atoms(
        numbers=np.asarray(numbers, dtype=np.int64),
        positions=np.asarray(pos, dtype=np.float64),
        cell=cell3(cell),
        pbc=True,
        tags=np.asarray(tags, dtype=np.int64).tolist(),
    )
    fixed = np.asarray(fixed, dtype=bool)
    if not fixed.any():
        fixed = np.asarray(tags, dtype=np.int64) == 0
    if fixed.any():
        atoms.set_constraint(FixAtoms(indices=np.where(fixed)[0].tolist()))
    return atoms


def relax_one(atoms: Atoms, calc, fmax: float, max_steps: int,
              lbfgs_memory: int, lbfgs_maxstep: float,
              lbfgs_damping: float, lbfgs_alpha: float) -> dict[str, Any]:
    atoms = atoms.copy()
    atoms.calc = calc
    energies: list[float] = []

    def record() -> None:
        try:
            energies.append(float(atoms.get_potential_energy()))
        except Exception:
            energies.append(float("nan"))

    record()
    opt = LBFGS(
        atoms,
        logfile=None,
        maxstep=float(lbfgs_maxstep),
        memory=int(lbfgs_memory),
        damping=float(lbfgs_damping),
        alpha=float(lbfgs_alpha),
    )
    opt.attach(record, interval=1)
    err = None
    try:
        converged = bool(opt.run(fmax=float(fmax), steps=int(max_steps)))
        forces = atoms.get_forces()
        fmax_val = float(np.linalg.norm(forces, axis=1).max())
    except Exception as exc:
        converged = False
        fmax_val = float("nan")
        err = repr(exc)
    return {
        "converged": bool(converged),
        "n_steps": int(getattr(opt, "nsteps", 0)),
        "fmax": fmax_val,
        "E_initial": float(energies[0]) if energies else float("nan"),
        "E_final": float(energies[-1]) if energies else float("nan"),
        "energies": [float(x) for x in energies],
        "error": err,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adsorbates-pkl", default="")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--refine-final", action="store_true")
    ap.add_argument("--prior-mode", default="random_heuristic",
                    choices=["random", "heuristic", "random_heuristic",
                             "harmonic_uniform", "harmonic_centered",
                             "catflow_center_rel", "gaussian_ads_train_stats"])
    ap.add_argument("--interstitial-gap", type=float, default=0.1)
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lbfgs-memory", type=int, default=100)
    ap.add_argument("--lbfgs-maxstep", type=float, default=0.2)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    adsorbates = Path(args.adsorbates_pkl or (repo / "data" / "pkls" / "adsorbates.pkl"))
    install_imports(repo, adsorbates)

    from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement
    from adsorbgen.evaluation.energy import UMAForce
    from adsorbgen.flow import euler_sample
    from adsorbgen.replay.eval import _model_cfg, _runtime_movable_mask
    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from geoopt.geoopt import load_model_from_ckpt

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    model, flow_cfg = load_model_from_ckpt(Path(args.ckpt), device)
    cfg = _model_cfg(model)
    use_ads_ref = bool(getattr(cfg, "use_ads_ref_pos", False))
    langevin_force_model = None
    if bool(getattr(cfg, "use_langevin_param", False)):
        langevin_force_model = UMAForce(
            model_name=str(getattr(cfg, "langevin_uma_model", "uma-s-1p2")),
            task_name=str(getattr(cfg, "langevin_uma_task", "oc20")),
            device=str(device),
        )
    slab_source = str(getattr(model, "adsorbgen_slab_source", "initial"))
    pristine_slabs = str(getattr(model, "adsorbgen_pristine_slabs", ""))
    pristine_index = str(getattr(model, "adsorbgen_pristine_index", ""))
    ds = PlacementPriorDataset(
        args.lmdb,
        prior_mode=args.prior_mode,
        interstitial_gap=args.interstitial_gap,
        adsorbates_pkl=str(adsorbates),
        max_samples=None,
        provide_ads_ref_pos=use_ads_ref,
        skip_anomaly=False,
        slab_source=slab_source,
        pristine_slabs=pristine_slabs,
        pristine_index=pristine_index,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    collate_fn=collate_displacement)
    predict_unit = pretrained_mlip.get_predict_unit(str(args.uma_model), device=str(device))
    calc = FAIRChemCalculator(predict_unit, task_name=str(args.uma_task))

    jobs: list[dict[str, Any]] = []
    for batch_i, batch in enumerate(dl):
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
            return out

        x_out = euler_sample(
            fwd,
            batch["pos"],
            movable,
            batch["pad_mask"],
            flow_cfg,
            num_steps=int(args.flow_steps),
            use_sde=False,
            refine_final=bool(args.refine_final),
        )
        for i in range(int(batch["pos"].shape[0])):
            n = int(batch["pad_mask"][i].sum().item())
            jobs.append({
                "idx": int(batch_i * int(args.batch_size) + i),
                "sid": int(batch["sid"][i].item()) if "sid" in batch else -1,
                "ads_id": int(batch["ads_id"][i].item()) if "ads_id" in batch else -1,
                "numbers": batch["atomic_numbers"][i, :n].detach().cpu().numpy().astype(np.int64),
                "tags": batch["tags"][i, :n].detach().cpu().numpy().astype(np.int64),
                "fixed": batch["fixed"][i, :n].detach().cpu().numpy().astype(np.int64),
                "cell": cell3(batch["cell"][i].detach().cpu().numpy()),
                "pos_prior": batch["pos"][i, :n].detach().cpu().numpy().astype(np.float64),
                "pos_pred": x_out[i, :n].detach().cpu().numpy().astype(np.float64),
                "pos_target": batch["pos_relaxed"][i, :n].detach().cpu().numpy().astype(np.float64),
            })

    rows: list[dict[str, Any]] = []
    for job in jobs:
        common = {
            k: job[k]
            for k in ["idx", "sid", "ads_id"]
        }
        for start_name, pos in [
            ("model_pred", job["pos_pred"]),
            ("random_prior", job["pos_prior"]),
            ("target_relaxed", job["pos_target"]),
        ]:
            atoms = atoms_from_arrays(job["numbers"], pos, job["cell"], job["tags"], job["fixed"])
            res = relax_one(
                atoms, calc,
                fmax=float(args.fmax),
                max_steps=int(args.max_steps),
                lbfgs_memory=int(args.lbfgs_memory),
                lbfgs_maxstep=float(args.lbfgs_maxstep),
                lbfgs_damping=float(args.lbfgs_damping),
                lbfgs_alpha=float(args.lbfgs_alpha),
            )
            rows.append({
                **common,
                "start": start_name,
                "n_atoms": int(len(job["numbers"])),
                "n_steps": int(res["n_steps"]),
                "converged": bool(res["converged"]),
                "fmax": float(res["fmax"]),
                "E_initial": float(res["E_initial"]),
                "E_final": float(res["E_final"]),
                "delta_E_relax": float(res["E_initial"] - res["E_final"]),
                "error": res["error"],
            })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_start = {}
    for start in sorted({r["start"] for r in rows}):
        rs = [r for r in rows if r["start"] == start]
        by_start[start] = {
            "n": len(rs),
            "convergence_rate": float(np.mean([r["converged"] for r in rs])) if rs else float("nan"),
            "n_steps_mean": float(np.mean([r["n_steps"] for r in rs])) if rs else float("nan"),
            "n_steps_median": float(np.median([r["n_steps"] for r in rs])) if rs else float("nan"),
            "E_initial_mean": float(np.mean([r["E_initial"] for r in rs])) if rs else float("nan"),
            "E_final_mean": float(np.mean([r["E_final"] for r in rs])) if rs else float("nan"),
            "delta_E_relax_mean": float(np.mean([r["delta_E_relax"] for r in rs])) if rs else float("nan"),
        }
    summary = {
        "ckpt": str(args.ckpt),
        "lmdb": str(args.lmdb),
        "flow_steps": int(args.flow_steps),
        "fmax": float(args.fmax),
        "max_steps": int(args.max_steps),
        "by_start": by_start,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
