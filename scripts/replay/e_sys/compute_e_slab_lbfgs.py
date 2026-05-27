#!/usr/bin/env python
"""Compute pristine slab-only references with ASE L-BFGS."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from tqdm.auto import tqdm


def _atoms_from_slab(rec: dict) -> Atoms:
    cell = np.asarray(rec["cell"], dtype=np.float64)
    if cell.ndim == 3:
        cell = cell[0]
    tags = np.asarray(rec.get("tags", np.zeros(len(rec["atomic_numbers"]))), dtype=np.int64)
    fixed = np.asarray(rec.get("fixed", np.zeros_like(tags)), dtype=np.int64).astype(bool)
    if not fixed.any():
        fixed = tags == 0
    atoms = Atoms(
        numbers=np.asarray(rec["atomic_numbers"], dtype=np.int64),
        positions=np.asarray(rec["pos"], dtype=np.float64),
        cell=cell.reshape(3, 3),
        pbc=True,
        tags=tags.tolist(),
    )
    if fixed.any():
        atoms.set_constraint(FixAtoms(indices=np.where(fixed)[0].tolist()))
    return atoms


def _relax(atoms: Atoms, calc, args: argparse.Namespace) -> dict:
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
        e_total = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        fmax = float(np.max(np.linalg.norm(forces, axis=1)))
        error = None
    except Exception as exc:
        converged = False
        e_total = float("nan")
        fmax = float("nan")
        error = repr(exc)
    return {
        "e_total": e_total,
        "E_slab_only": e_total,
        "converged": converged,
        "fmax": fmax,
        "n_steps": int(getattr(opt, "nsteps", 0)),
        "n_atoms": len(atoms),
        "pos_relaxed": atoms.get_positions().astype(np.float32),
        "cell": np.asarray(atoms.cell.array, dtype=np.float32),
        "atomic_numbers": np.asarray(atoms.numbers, dtype=np.int64),
        "tags": np.asarray(atoms.get_tags(), dtype=np.int64),
        "error": error,
        "relaxer": "ase.LBFGS",
        "uma_model": args.uma_model,
        "uma_task": args.uma_task,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pristine-slabs", default="/home/irteam/results/pristine_slabs/is2res.pkl")
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=24)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--uma-fmax", type=float, default=0.05)
    ap.add_argument("--uma-max-steps", type=int, default=300)
    ap.add_argument("--lbfgs-maxstep", type=float, default=0.04)
    ap.add_argument("--lbfgs-memory", type=int, default=50)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"e_slab_only_lbfgs_shard{args.shard_idx}.pkl"

    with open(args.pristine_slabs, "rb") as f:
        slabs = pickle.load(f)
    keys = sorted(slabs.keys(), key=repr)
    mine = keys[args.shard_idx::args.num_shards]
    print(f"[slab shard{args.shard_idx}] assigned {len(mine)} / {len(keys)} slabs", flush=True)

    results: dict[tuple, dict] = {}
    if args.resume and out_path.exists():
        with out_path.open("rb") as f:
            results = pickle.load(f)
        print(f"[slab shard{args.shard_idx}] resume loaded {len(results)} slabs", flush=True)

    predict_unit = pretrained_mlip.get_predict_unit(args.uma_model, device=str(device))
    calc = FAIRChemCalculator(predict_unit, task_name=args.uma_task)

    t0 = time.time()
    pbar = tqdm(mine, desc=f"[slab shard{args.shard_idx}] slab L-BFGS", dynamic_ncols=True)
    for n_done, key in enumerate(pbar, start=1):
        if key in results:
            continue
        atoms = _atoms_from_slab(slabs[key])
        rec = _relax(atoms, calc, args)
        rec["slab_key"] = key
        results[key] = rec
        elapsed = max(time.time() - t0, 1e-6)
        n_conv = sum(1 for r in results.values() if r.get("converged"))
        pbar.set_postfix(records=len(results), conv=n_conv, rate=f"{len(results)/elapsed:.2f}/s")
        if n_done % max(args.save_every, 1) == 0:
            with out_path.open("wb") as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    with out_path.open("wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.time() - t0
    print(f"[slab shard{args.shard_idx}] DONE {len(results)} slabs in {elapsed:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
