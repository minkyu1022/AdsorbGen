#!/usr/bin/env python
"""Pilot UMA-s-1p2 re-relaxation from existing relaxed geometries.

Samples records from current ID train/val, bare-slab cache, and OC20-dense,
then runs ASE LBFGS from the stored relaxed positions.  The goal is to estimate
how expensive a full 1p1 -> 1p2 reference relabel will be before launching it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from tqdm.auto import tqdm


def _open_lmdb(path: str) -> lmdb.Environment:
    try:
        return lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False)
    except lmdb.Error:
        return lmdb.open(path, readonly=True, lock=False, readahead=False)


def _lmdb_len(env: lmdb.Environment) -> int:
    with env.begin() as txn:
        raw = txn.get(b"length")
        if raw is not None:
            return int(pickle.loads(raw))
        return int(txn.stat()["entries"])


def _read_lmdb(env: lmdb.Environment, idx: int) -> dict:
    with env.begin() as txn:
        raw = txn.get(str(int(idx)).encode("ascii"))
    if raw is None:
        raise KeyError(idx)
    return pickle.loads(raw)


def _cell(entry: dict) -> np.ndarray:
    cell = np.asarray(entry["cell"], dtype=np.float64)
    if cell.ndim == 3:
        cell = cell[0]
    return cell.reshape(3, 3)


def _fixed(entry: dict, tags: np.ndarray) -> np.ndarray:
    fixed = np.asarray(entry.get("fixed", np.zeros_like(tags)), dtype=np.int64).astype(bool)
    if not fixed.any():
        fixed = tags == 0
    return fixed


def _atoms_from_entry(entry: dict, start_key: str = "pos_relaxed") -> Atoms:
    tags = np.asarray(entry.get("tags", np.zeros(len(entry["atomic_numbers"]))), dtype=np.int64)
    fixed = _fixed(entry, tags)
    pos = np.asarray(entry[start_key], dtype=np.float64)
    atoms = Atoms(
        numbers=np.asarray(entry["atomic_numbers"], dtype=np.int64),
        positions=pos,
        cell=_cell(entry),
        pbc=True,
        tags=tags.tolist(),
    )
    if fixed.any():
        atoms.set_constraint(FixAtoms(indices=np.where(fixed)[0].tolist()))
    return atoms


def _atoms_from_slab(rec: dict) -> Atoms:
    tags = np.asarray(rec.get("tags", np.zeros(len(rec["atomic_numbers"]))), dtype=np.int64)
    fixed = _fixed(rec, tags)
    start_key = "pos_relaxed" if "pos_relaxed" in rec else "pos"
    atoms = Atoms(
        numbers=np.asarray(rec["atomic_numbers"], dtype=np.int64),
        positions=np.asarray(rec[start_key], dtype=np.float64),
        cell=_cell(rec),
        pbc=True,
        tags=tags.tolist(),
    )
    if fixed.any():
        atoms.set_constraint(FixAtoms(indices=np.where(fixed)[0].tolist()))
    return atoms


def _relax(atoms: Atoms, calc, args: argparse.Namespace) -> dict:
    start_pos = atoms.get_positions().copy()
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
        converged = bool(opt.run(fmax=args.fmax, steps=args.max_steps))
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        fmax = float(np.max(np.linalg.norm(forces, axis=1)))
        error = None
    except Exception as exc:
        converged = False
        energy = float("nan")
        fmax = float("nan")
        error = repr(exc)
    disp = atoms.get_positions() - start_pos
    return {
        "converged": converged,
        "n_steps": int(getattr(opt, "nsteps", 0)),
        "e_total": energy,
        "fmax": fmax,
        "rms_disp_A": float(np.sqrt(np.mean(np.sum(disp * disp, axis=1)))),
        "max_disp_A": float(np.sqrt(np.sum(disp * disp, axis=1)).max()),
        "n_atoms": int(len(atoms)),
        "error": error,
    }


def _sample_indices(n: int, k: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    return sorted(idxs[: min(k, n)])


def _build_jobs(args: argparse.Namespace) -> list[tuple[str, str, object]]:
    jobs: list[tuple[str, str, object]] = []

    for lmdb_path in args.id_lmdbs:
        env = _open_lmdb(lmdb_path)
        n = _lmdb_len(env)
        for idx in _sample_indices(n, args.sample_id_per_lmdb, args.seed + abs(hash(lmdb_path)) % 100000):
            jobs.append(("id_adslab", lmdb_path, idx))
        env.close()

    with open(args.bare_slab_pkl, "rb") as f:
        slabs = pickle.load(f)
    slab_keys = sorted(slabs.keys(), key=repr)
    rng = random.Random(args.seed + 17)
    rng.shuffle(slab_keys)
    for key in slab_keys[: min(args.sample_bare_slab, len(slab_keys))]:
        jobs.append(("bare_slab", args.bare_slab_pkl, key))

    env = _open_lmdb(args.oc20dense_lmdb)
    n_dense = _lmdb_len(env)
    for idx in _sample_indices(n_dense, args.sample_oc20dense, args.seed + 29):
        jobs.append(("oc20dense", args.oc20dense_lmdb, idx))
    env.close()

    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-lmdbs", nargs="+", default=[
        "/home1/irteam/data/processed_ID/is2res_train.lmdb",
        "/home1/irteam/data/processed_ID/is2res_val.lmdb",
    ])
    ap.add_argument("--bare-slab-pkl", default="/home1/irteam/data-vol1/minkyu/data/replay/E_slab_only_lbfgs_by_slab.pkl")
    ap.add_argument("--oc20dense-lmdb", default="/home1/irteam/data/processed_old/oc20dense.lmdb")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--sample-id-per-lmdb", type=int, default=512)
    ap.add_argument("--sample-bare-slab", type=int, default=1024)
    ap.add_argument("--sample-oc20dense", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260619)
    ap.add_argument("--uma-model", default="uma-s-1p2")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lbfgs-maxstep", type=float, default=0.2)
    ap.add_argument("--lbfgs-memory", type=int, default=100)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pilot_shard{args.shard_idx}.pkl"
    summary_path = out_dir / f"pilot_shard{args.shard_idx}.json"
    results = {}
    if args.resume and out_path.exists():
        with out_path.open("rb") as f:
            results = pickle.load(f)

    jobs = _build_jobs(args)
    mine = jobs[args.shard_idx :: args.num_shards]
    print(f"[pilot {args.shard_idx}] assigned {len(mine)} / {len(jobs)} jobs", flush=True)

    env_cache: dict[str, lmdb.Environment] = {}
    pkl_cache: dict[str, dict] = {}
    predict_unit = pretrained_mlip.get_predict_unit(args.uma_model, device="cuda")
    calc = FAIRChemCalculator(predict_unit, task_name=args.uma_task)

    t0 = time.time()
    for i, (kind, src, key) in enumerate(tqdm(mine, desc=f"pilot shard {args.shard_idx}", dynamic_ncols=True), start=1):
        rid = f"{kind}:{src}:{key!r}"
        if rid in results:
            continue
        try:
            if kind in {"id_adslab", "oc20dense"}:
                env = env_cache.get(src)
                if env is None:
                    env = _open_lmdb(src)
                    env_cache[src] = env
                entry = _read_lmdb(env, int(key))
                atoms = _atoms_from_entry(entry, "pos_relaxed")
                meta = {
                    "system_key": entry.get("system_key"),
                    "config_key": entry.get("config_key"),
                    "sid": entry.get("sid"),
                }
            else:
                slabs = pkl_cache.get(src)
                if slabs is None:
                    with open(src, "rb") as f:
                        slabs = pickle.load(f)
                    pkl_cache[src] = slabs
                rec = slabs[key]
                atoms = _atoms_from_slab(rec)
                meta = {"slab_key": repr(key)}
            rec_out = _relax(atoms, calc, args)
            rec_out.update({"kind": kind, "source": src, "key": repr(key), **meta})
        except Exception as exc:
            rec_out = {"kind": kind, "source": src, "key": repr(key), "converged": False, "error": repr(exc)}
        results[rid] = rec_out
        if i % max(args.save_every, 1) == 0:
            with out_path.open("wb") as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    with out_path.open("wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    for env in env_cache.values():
        env.close()

    by_kind = {}
    for kind in sorted({r["kind"] for r in results.values()}):
        vals = [r for r in results.values() if r["kind"] == kind]
        steps = np.asarray([r.get("n_steps", np.nan) for r in vals], dtype=float)
        conv = np.asarray([bool(r.get("converged", False)) for r in vals])
        by_kind[kind] = {
            "n": len(vals),
            "converged": int(conv.sum()),
            "converged_rate": float(conv.mean()) if len(conv) else 0.0,
            "steps_mean": float(np.nanmean(steps)) if len(steps) else None,
            "steps_median": float(np.nanmedian(steps)) if len(steps) else None,
            "steps_p95": float(np.nanpercentile(steps, 95)) if len(steps) else None,
            "steps_max": float(np.nanmax(steps)) if len(steps) else None,
        }
    summary = {
        "uma_model": args.uma_model,
        "uma_task": args.uma_task,
        "fmax": args.fmax,
        "max_steps": args.max_steps,
        "lbfgs_maxstep": args.lbfgs_maxstep,
        "lbfgs_memory": args.lbfgs_memory,
        "elapsed_sec": time.time() - t0,
        "by_kind": by_kind,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
