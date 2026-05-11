#!/usr/bin/env python
"""UMA-relax extracted OC20-Dense clean slabs on one or more GPUs.

Input is the compact DB produced by ``extract_oc20dense_pristine_slabs.py``:

    {slab_key: {
        "system_key": str,
        "pos": (N, 3),
        "atomic_numbers": (N,),
        "cell": (3, 3),
        ...
    }}

The Dense processed LMDB is scanned once to recover slab tags/fixed masks for
each ``system_key``. During relaxation, bulk/fixed atoms are constrained and
surface atoms are allowed to move. Output is compatible with
``adsorbgen.eval.load_pristine_context`` when paired with the original
``oc20dense.system_index.pkl``.
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import LBFGS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_system_slab_masks(processed_lmdb: Path) -> dict[str, dict[str, np.ndarray]]:
    """Return first slab-only tags/fixed/atomic_numbers record per system_key."""
    out: dict[str, dict[str, np.ndarray]] = {}
    env = lmdb.open(str(processed_lmdb), subdir=False, readonly=True, lock=False)
    try:
        with env.begin() as txn:
            raw_n = txn.get(b"length")
            n_total = int(pickle.loads(raw_n)) if raw_n is not None else txn.stat()["entries"]
            for i in range(n_total):
                raw = txn.get(str(i).encode("ascii"))
                if raw is None:
                    continue
                rec = pickle.loads(raw)
                system_key = str(rec.get("system_key", ""))
                if not system_key or system_key in out:
                    continue
                tags = to_numpy(rec["tags"]).astype(np.int64)
                slab_mask = tags != 2
                fixed = to_numpy(rec.get("fixed", np.zeros_like(tags))).astype(bool)
                out[system_key] = {
                    "tags": tags[slab_mask],
                    "fixed": fixed[slab_mask],
                    "atomic_numbers": to_numpy(rec["atomic_numbers"])[slab_mask].astype(np.int64),
                }
    finally:
        env.close()
    log.info("Loaded slab masks for %d Dense systems from %s", len(out), processed_lmdb)
    return out


def attach_atoms_and_constraints(db: dict, system_masks: dict[str, dict[str, np.ndarray]]) -> tuple[dict, dict]:
    """Build an in-memory relaxation cache keyed by slab_key."""
    slabs = {}
    stats = {"records": 0, "with_masks": 0, "number_mismatch": 0, "unconstrained": 0}
    for slab_key, rec in db.items():
        stats["records"] += 1
        system_key = str(rec.get("system_key", ""))
        numbers = np.asarray(rec["atomic_numbers"], dtype=np.int64)
        pos = np.asarray(rec["pos"], dtype=np.float64)
        cell = np.asarray(rec["cell"], dtype=np.float64)
        tags = np.ones(len(numbers), dtype=np.int64)
        fixed = np.zeros(len(numbers), dtype=bool)

        mask_rec = system_masks.get(system_key)
        if mask_rec is not None and len(mask_rec["atomic_numbers"]) == len(numbers):
            stats["with_masks"] += 1
            tags = mask_rec["tags"].astype(np.int64, copy=True)
            fixed = mask_rec["fixed"].astype(bool, copy=True)
            if not np.array_equal(mask_rec["atomic_numbers"], numbers):
                stats["number_mismatch"] += 1
        else:
            stats["unconstrained"] += 1

        atoms = Atoms(numbers=numbers.astype(int), positions=pos, cell=cell, pbc=True)
        fix_indices = np.where((tags == 0) | fixed)[0]
        if len(fix_indices) > 0:
            atoms.set_constraint(FixAtoms(indices=fix_indices))

        area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
        slabs[slab_key] = {
            "atoms": atoms,
            "system_key": system_key,
            "slab_key": rec.get("slab_key", slab_key),
            "atomic_numbers": numbers,
            "tags": tags,
            "fixed": fixed,
            "cell": cell,
            "pbc": np.asarray(rec.get("pbc", [True, True, True]), dtype=bool),
            "n_atoms": int(len(numbers)),
            "area": area,
        }
    return slabs, stats


def _make_clean_record(slab_info: dict, atoms_after, e_total: float | None,
                       forces_max: float | None, converged: bool,
                       n_steps: int, error: str | None) -> dict:
    return {
        "system_key": slab_info["system_key"],
        "slab_key": slab_info["slab_key"],
        "pos": np.asarray(atoms_after.positions, dtype=np.float64).copy(),
        "atomic_numbers": np.asarray(slab_info["atomic_numbers"], dtype=np.int64),
        "tags": np.asarray(slab_info["tags"], dtype=np.int64),
        "fixed": np.asarray(slab_info["fixed"], dtype=bool),
        "cell": np.asarray(slab_info["cell"], dtype=np.float64),
        "pbc": np.asarray(slab_info["pbc"], dtype=bool),
        "n_atoms": int(slab_info["n_atoms"]),
        "area": float(slab_info["area"]),
        "e_total": float(e_total) if e_total is not None and np.isfinite(e_total) else float("nan"),
        "forces_max": float(forces_max) if forces_max is not None and np.isfinite(forces_max) else float("nan"),
        "converged": bool(converged),
        "n_steps": int(n_steps),
        "error": error,
    }


def _relax_one(item, *, batch_predict_unit=None, calc=None, task: str = "oc20",
               fmax: float = 0.05, max_steps: int = 300):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    slab_key, slab_info = item
    atoms = slab_info["atoms"].copy()
    if batch_predict_unit is not None:
        atoms.calc = FAIRChemCalculator(batch_predict_unit, task_name=task)
    else:
        atoms.calc = calc

    opt = LBFGS(atoms, logfile=None)
    try:
        converged = opt.run(fmax=fmax, steps=max_steps)
        e_total = atoms.get_potential_energy()
        forces = atoms.get_forces()
        f_max = float(np.max(np.linalg.norm(forces, axis=1)))
        rec = _make_clean_record(slab_info, atoms, e_total, f_max, converged, opt.nsteps, None)
    except Exception as exc:
        rec = _make_clean_record(slab_info, atoms, None, None, False, opt.nsteps, str(exc))
    atoms.calc = None
    return slab_key, rec


def gpu_worker(gpu_id, slab_cache_path, slab_keys, model_name, task, fmax,
               max_steps, num_workers, results_dir, checkpoint_every):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["RAY_TMPDIR"] = f"/tmp/ray_oc20dense_slab_gpu{gpu_id}"

    import ray.serve as _ray_serve
    from ray.serve.config import HTTPOptions as _HTTPOptions
    _orig_start = _ray_serve.start

    def _patched_start(*args, **kwargs):
        kwargs.setdefault("http_options", _HTTPOptions(location="NoServer"))
        return _orig_start(*args, **kwargs)

    _ray_serve.start = _patched_start

    with open(slab_cache_path, "rb") as f:
        all_slabs = pickle.load(f)
    my_items = [(k, all_slabs[k]) for k in slab_keys]
    del all_slabs

    print(f"[GPU {gpu_id}] Loading model {model_name}...", flush=True)
    from fairchem.core.calculate.pretrained_mlip import get_predict_unit
    from fairchem.core.calculate import InferenceBatcher

    predict_unit = get_predict_unit(model_name, device="cuda")
    batcher = InferenceBatcher(
        predict_unit,
        concurrency_backend_options={"max_workers": num_workers},
    )
    process_fn = functools.partial(
        _relax_one,
        batch_predict_unit=batcher.batch_predict_unit,
        task=task,
        fmax=fmax,
        max_steps=max_steps,
    )

    results = {}
    checkpoint_path = Path(results_dir) / f"oc20dense_relax_checkpoint_gpu{gpu_id}.pkl"
    t_start = time.time()
    total = len(my_items)
    print(f"[GPU {gpu_id}] Processing {total} slabs (workers={num_workers})...", flush=True)

    for i, (slab_key, rec) in enumerate(batcher.executor.map(process_fn, my_items)):
        results[slab_key] = rec
        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (total - i - 1) / rate if rate > 0 else 0
            conv = sum(1 for r in results.values() if r["converged"])
            print(
                f"[GPU {gpu_id}] [{i+1}/{total}] conv={conv}/{i+1} "
                f"rate={rate:.2f}/s ETA={eta/60:.1f}m",
                flush=True,
            )
        if (i + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(checkpoint_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[GPU {gpu_id}] Done. {len(results)} results.", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-pkl", type=Path, required=True)
    p.add_argument("--processed-lmdb", type=Path, default=Path("data/processed/oc20dense.lmdb"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu-ids", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--model", default="uma-s-1p1")
    p.add_argument("--task", default="oc20")
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--max-slabs", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=100)
    args = p.parse_args()

    if not args.in_pkl.exists():
        log.error("missing input pkl: %s", args.in_pkl)
        sys.exit(1)
    if not args.processed_lmdb.exists():
        log.error("missing processed LMDB: %s", args.processed_lmdb)
        sys.exit(1)

    with open(args.in_pkl, "rb") as f:
        db = pickle.load(f)
    if args.max_slabs is not None:
        db = dict(list(db.items())[:args.max_slabs])
    masks = load_system_slab_masks(args.processed_lmdb)
    slabs, stats = attach_atoms_and_constraints(db, masks)
    log.info("Prepared %d slabs: %s", len(slabs), stats)
    if stats["unconstrained"]:
        log.warning("%d slabs had no processed-LMDB mask and will relax unconstrained", stats["unconstrained"])

    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    items = list(slabs.items())

    if args.cpu or args.gpu_ids is None:
        from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
        from fairchem.core.calculate.pretrained_mlip import get_predict_unit

        device = "cpu" if args.cpu else "cuda"
        log.info("Loading UMA model %s on %s", args.model, device)
        predict_unit = get_predict_unit(args.model, device=device)
        calc = FAIRChemCalculator(predict_unit, task_name=args.task)
        results = {}
        for i, item in enumerate(items):
            slab_key, rec = _relax_one(
                item, calc=calc, task=args.task, fmax=args.fmax, max_steps=args.max_steps
            )
            results[slab_key] = rec
            log.info("[%d/%d] converged=%s steps=%d fmax=%.4f",
                     i + 1, len(items), rec["converged"], rec["n_steps"], rec["forces_max"])
        with open(args.out, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Wrote %d records -> %s", len(results), args.out)
        return

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    log.info("Multi-GPU mode: GPUs %s, %d workers/GPU", gpu_ids, args.num_workers)
    cache_path = args.out.with_suffix(".cache.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(slabs, f, protocol=pickle.HIGHEST_PROTOCOL)

    keys_per_gpu: list[list] = [[] for _ in gpu_ids]
    for i, (k, _v) in enumerate(items):
        keys_per_gpu[i % len(gpu_ids)].append(k)
    for idx, gid in enumerate(gpu_ids):
        log.info("  GPU %s: %d slabs", gid, len(keys_per_gpu[idx]))

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = []
    for idx, gid in enumerate(gpu_ids):
        proc = ctx.Process(
            target=gpu_worker,
            args=(
                gid,
                str(cache_path),
                keys_per_gpu[idx],
                args.model,
                args.task,
                args.fmax,
                args.max_steps,
                args.num_workers,
                str(args.out.parent),
                args.checkpoint_every,
            ),
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()

    merged = {}
    for gid in gpu_ids:
        cp = args.out.parent / f"oc20dense_relax_checkpoint_gpu{gid}.pkl"
        if cp.exists():
            with open(cp, "rb") as f:
                merged.update(pickle.load(f))
            log.info("merged %s", cp.name)

    with open(args.out, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    conv = sum(1 for r in merged.values() if r["converged"])
    log.info("Wrote %d records -> %s (converged %d/%d)", len(merged), args.out, conv, len(merged))


if __name__ == "__main__":
    main()
