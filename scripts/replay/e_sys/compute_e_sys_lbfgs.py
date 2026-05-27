#!/usr/bin/env python
"""Compute UMA/OC20 E_sys references with ASE L-BFGS.

This is the L-BFGS counterpart of ``compute_e_sys.py``.  It reads clean
train/val-ID LMDB rows, starts from each row's ``pos_relaxed`` geometry, relaxes
with the same ASE L-BFGS settings used by the self-improvement replay, and
writes one shard dictionary keyed by sid.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from tqdm.auto import tqdm


def _clean_indices(lmdb_path: str) -> list[int]:
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        raw_len = txn.get(b"length")
        n = int(pickle.loads(raw_len)) if raw_len is not None else int(txn.stat()["entries"])
        mask_raw = txn.get(b"anomaly_mask")
    env.close()
    if mask_raw is None:
        return list(range(n))
    mask = np.asarray(pickle.loads(mask_raw), dtype=np.int8)[:n]
    return np.where(mask == 0)[0].tolist()


def _read_entry(env: lmdb.Environment, idx: int) -> dict:
    with env.begin() as txn:
        raw = txn.get(str(int(idx)).encode("ascii"))
    if raw is None:
        raise KeyError(f"missing LMDB row {idx}")
    return pickle.loads(raw)


def _atoms_from_entry(entry: dict) -> Atoms:
    cell = np.asarray(entry["cell"], dtype=np.float64)
    if cell.ndim == 3:
        cell = cell[0]
    tags = np.asarray(entry["tags"], dtype=np.int64)
    fixed = np.asarray(entry.get("fixed", np.zeros_like(tags)), dtype=np.int64).astype(bool)
    if not fixed.any():
        fixed = tags == 0
    atoms = Atoms(
        numbers=np.asarray(entry["atomic_numbers"], dtype=np.int64),
        positions=np.asarray(entry["pos_relaxed"], dtype=np.float64),
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
    except Exception as exc:  # keep partial results robust across bad systems
        converged = False
        e_total = float("nan")
        fmax = float("nan")
        error = repr(exc)
    return {
        "e_total": e_total,
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
    ap.add_argument("--lmdbs", nargs="+", required=True)
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
    out_path = out_dir / f"e_sys_lbfgs_shard{args.shard_idx}.pkl"

    work_keys: list[tuple[int, int]] = []
    for lid, path in enumerate(args.lmdbs):
        idxs = _clean_indices(path)
        work_keys.extend((lid, i) for i in idxs)
        print(f"[shard{args.shard_idx}] {Path(path).name}: {len(idxs)} clean", flush=True)
    mine = work_keys[args.shard_idx::args.num_shards]
    print(f"[shard{args.shard_idx}] assigned {len(mine)} / {len(work_keys)} rows", flush=True)

    results: dict[int, dict] = {}
    if args.resume and out_path.exists():
        with out_path.open("rb") as f:
            results = pickle.load(f)
        print(f"[shard{args.shard_idx}] resume loaded {len(results)} sids", flush=True)

    envs = [
        lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False)
        for path in args.lmdbs
    ]
    predict_unit = pretrained_mlip.get_predict_unit(args.uma_model, device=str(device))
    calc = FAIRChemCalculator(predict_unit, task_name=args.uma_task)

    t0 = time.time()
    pbar = tqdm(mine, desc=f"[shard{args.shard_idx}] E_sys L-BFGS", dynamic_ncols=True)
    for n_done, (lid, idx) in enumerate(pbar, start=1):
        entry = _read_entry(envs[int(lid)], int(idx))
        sid = int(entry["sid"])
        if sid in results:
            continue
        atoms = _atoms_from_entry(entry)
        rec = _relax(atoms, calc, args)
        rec.update({"sid": sid, "lmdb_id": int(lid), "raw_idx": int(idx)})
        results[sid] = rec
        elapsed = max(time.time() - t0, 1e-6)
        n_conv = sum(1 for r in results.values() if r.get("converged"))
        pbar.set_postfix(records=len(results), conv=n_conv, rate=f"{len(results)/elapsed:.2f}/s")
        if n_done % max(args.save_every, 1) == 0:
            with out_path.open("wb") as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    with out_path.open("wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    for env in envs:
        env.close()
    elapsed = time.time() - t0
    print(f"[shard{args.shard_idx}] DONE {len(results)} records in {elapsed:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
