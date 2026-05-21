#!/usr/bin/env python
"""Compute UMA-s-1p1 / task=oc20 relaxed system energies (E_sys), one shard/GPU.

For every clean OC20 training entry, re-relaxes the dataset's DFT-relaxed
geometry (LMDB ``pos_relaxed``) with UMA FIRE — OC20 ``fixed`` atoms frozen —
to fmax convergence, and records the relaxed UMA oc20 energy. The dataset
structures are DFT minima; this re-relaxation puts them at the MLIP's own
minimum so the energy is consistent with the replay daemon's E_sys_pred
(same model / task / relaxer).

Output per shard: ``{out_dir}/e_sys_shard{N}.pkl`` — dict sid -> {e_total,
converged, fmax, n_steps, n_atoms, pos_relaxed}. Merge shards into the final
E_sys.pkl.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm.auto import tqdm

_REPO = Path(__file__).resolve().parents[1]
if (_REPO / "adsorbgen").is_dir():
    sys.path.insert(0, str(_REPO))

from adsorbgen.replay.viz import FixedAtomsHook  # noqa: E402
from adsorbgen.replay.eval import _chunk_by_atom_budget  # noqa: E402


class StepCounterHook:
    """Count FIRE post-update steps without storing trajectories."""

    def __init__(self):
        from nvalchemi.dynamics import DynamicsStage

        self.frequency = 1
        self.stage = DynamicsStage.AFTER_STEP
        self.n_steps = 0

    def __call__(self, ctx, stage) -> None:  # noqa: ARG002
        self.n_steps += 1


def _clean_indices(lmdb_path: str):
    """Raw entry indices with anomaly_mask == 0 (clean), matching training."""
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        n = int(pickle.loads(txn.get(b"length")))
        mask_raw = txn.get(b"anomaly_mask")
    env.close()
    if mask_raw is None:
        return list(range(n))
    mask = np.asarray(pickle.loads(mask_raw), dtype=np.int8)[:n]
    return np.where(mask == 0)[0].tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdbs", nargs="+", required=True,
                   help="training LMDB paths (clean entries only are used)")
    p.add_argument("--shard-idx", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--uma-model", default="uma-s-1p1")
    p.add_argument("--uma-task", default="oc20")
    p.add_argument("--uma-fmax", type=float, default=0.05)
    p.add_argument("--uma-max-steps", type=int, default=200)
    p.add_argument("--uma-fire-dt", type=float, default=0.02)
    p.add_argument("--atom-budget", type=int, default=4000)
    p.add_argument("--save-every-chunks", type=int, default=40)
    p.add_argument("--max-systems", type=int, default=0,
                   help="debug/smoke-test limit after sharding; 0 = all")
    p.add_argument("--resume", action="store_true",
                   help="load an existing shard pkl and skip completed sids")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    from fast_dynamics import UMAWrapper, prepare_batch_for_dynamics
    from nvalchemi.data import AtomicData as NVAtomicData
    from nvalchemi.data import Batch as NVBatch
    from nvalchemi.dynamics import FIRE, ConvergenceHook

    # ---- global clean (lmdb_id, raw_idx) list; this shard takes a stride slice
    work_keys = []
    for lid, path in enumerate(args.lmdbs):
        idxs = _clean_indices(path)
        work_keys.extend((lid, i) for i in idxs)
        print(f"[shard{args.shard_idx}] {Path(path).name}: {len(idxs)} clean",
              flush=True)
    mine = work_keys[args.shard_idx::args.num_shards]
    if args.max_systems and args.max_systems > 0:
        mine = mine[:args.max_systems]
    print(f"[shard{args.shard_idx}] assigned {len(mine)} / {len(work_keys)} systems",
          flush=True)

    uma = UMAWrapper.from_checkpoint(args.uma_model, task_name=args.uma_task,
                                     device=device)

    envs = {lid: lmdb.open(path, subdir=False, readonly=True, lock=False)
            for lid, path in enumerate(args.lmdbs)}
    txns = {lid: e.begin() for lid, e in envs.items()}

    def load(lid, idx):
        e = pickle.loads(txns[lid].get(str(idx).encode()))
        cell = np.asarray(e["cell"], dtype=np.float32)
        if cell.ndim == 3:
            cell = cell[0]
        return dict(
            sid=int(e["sid"]),
            numbers=np.asarray(e["atomic_numbers"], dtype=np.int64),
            pos=np.asarray(e["pos_relaxed"], dtype=np.float64),
            cell=cell.reshape(3, 3),
            tags=np.asarray(e["tags"], dtype=np.int64),
            fixed=np.asarray(e["fixed"], dtype=np.int64),
            n_atoms=int(len(e["atomic_numbers"])),
        )

    out_path = Path(args.out_dir) / f"e_sys_shard{args.shard_idx}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    if args.resume and out_path.exists():
        with open(out_path, "rb") as f:
            results = pickle.load(f)
        print(f"[shard{args.shard_idx}] resume: loaded {len(results)} existing sids",
              flush=True)

    items = [load(lid, idx) for lid, idx in mine]
    if results:
        items = [it for it in items if int(it["sid"]) not in results]
        print(f"[shard{args.shard_idx}] remaining after resume: {len(items)} systems",
              flush=True)
    chunks = _chunk_by_atom_budget(items, args.atom_budget)

    t0 = time.time()
    pbar = tqdm(chunks, desc=f"[shard{args.shard_idx}] E_sys relax",
                unit="chunk", dynamic_ncols=True)
    for ci, chunk in enumerate(pbar):
        finite = [it for it in chunk if np.isfinite(it["pos"]).all()]
        if not finite:
            continue
        data_list = [
            NVAtomicData(
                positions=torch.as_tensor(it["pos"], dtype=torch.float32, device=device),
                atomic_numbers=torch.as_tensor(it["numbers"], dtype=torch.long, device=device),
                cell=torch.as_tensor(it["cell"], dtype=torch.float32,
                                     device=device).reshape(1, 3, 3),
                pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
            )
            for it in finite
        ]
        nvbatch = NVBatch.from_data_list(data_list)
        prepare_batch_for_dynamics(nvbatch)

        fixed_parts = []
        for it in finite:
            fm = it["fixed"].astype(bool)
            if not fm.any():
                fm = it["tags"] == 0
            fixed_parts.append(fm)
        fixed_mask = torch.as_tensor(np.concatenate(fixed_parts), dtype=torch.bool,
                                     device=device)

        fire = FIRE(uma, dt=args.uma_fire_dt, n_steps=args.uma_max_steps,
                    convergence_hook=ConvergenceHook.from_fmax(args.uma_fmax))
        if fixed_mask.any():
            fire.register_hook(FixedAtomsHook(fixed_mask))
        step_counter = StepCounterHook()
        fire.register_hook(step_counter)
        fire.run(nvbatch)

        ptr = nvbatch.batch_ptr.long().tolist()
        energies = nvbatch.energy.detach().squeeze(-1).float().cpu().tolist()
        forces = nvbatch.forces.detach()
        positions_out = nvbatch.positions.detach()
        for i, it in enumerate(finite):
            s, e_ = ptr[i], ptr[i + 1]
            fmax_i = float(forces[s:e_].norm(dim=-1).max().item())
            results[it["sid"]] = dict(
                e_total=float(energies[i]),
                converged=bool(fmax_i <= args.uma_fmax),
                fmax=fmax_i,
                n_steps=int(step_counter.n_steps),
                n_atoms=it["n_atoms"],
                pos_relaxed=positions_out[s:e_].cpu().numpy().astype(np.float32),
            )
        n_conv = sum(1 for r in results.values() if r["converged"])
        pbar.set_postfix(systems=len(results), converged=n_conv,
                         rate=f"{len(results)/(time.time()-t0):.2f}/s")
        del nvbatch, forces, positions_out, fire
        torch.cuda.empty_cache()
        if (ci + 1) % args.save_every_chunks == 0:
            with open(out_path, "wb") as f:
                pickle.dump(results, f)

    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    el = time.time() - t0
    print(f"[shard{args.shard_idx}] DONE: {len(results)} systems in {el:.0f}s "
          f"({len(results)/el:.2f} sys/s) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
