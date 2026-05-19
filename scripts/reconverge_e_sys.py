#!/usr/bin/env python
"""Re-relax the non-converged E_sys entries with a higher FIRE step cap.

``compute_e_sys.py`` relaxes every clean OC20 entry with a 200-step cap; ~9%
do not reach fmax convergence in that budget. This script loads an existing
``e_sys_shard{N}.pkl``, takes only the records with ``converged == False``,
and *continues* their relaxation from the stored (partially-relaxed) geometry
with a larger step cap. The shard pkl is updated in place.

Output: the same ``{shards_dir}/e_sys_shard{N}.pkl`` — non-converged records
get fresh {e_total, converged, fmax, n_steps, pos_relaxed}; converged records
are left untouched.
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

from adsorbgen.replay_viz import FixedAtomsHook  # noqa: E402
from adsorbgen.eval_replay import _chunk_by_atom_budget  # noqa: E402


def _clean_indices(lmdb_path: str):
    """Raw entry indices with anomaly_mask == 0 (clean) — matches compute_e_sys."""
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        n = int(pickle.loads(txn.get(b"length")))
        mask_raw = txn.get(b"anomaly_mask")
    env.close()
    if mask_raw is None:
        return list(range(n))
    mask = np.asarray(pickle.loads(mask_raw), dtype=np.int8)[:n]
    return np.where(mask == 0)[0].tolist()


class StepCounterHook:
    """Count FIRE post-update steps without storing trajectories."""

    def __init__(self):
        from nvalchemi.dynamics import DynamicsStage

        self.frequency = 1
        self.stage = DynamicsStage.AFTER_STEP
        self.n_steps = 0

    def __call__(self, ctx, stage) -> None:  # noqa: ARG002
        self.n_steps += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdbs", nargs="+", required=True,
                   help="same training LMDB list passed to compute_e_sys.py")
    p.add_argument("--shard-idx", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--shards-dir", required=True,
                   help="directory holding e_sys_shard{N}.pkl (updated in place)")
    p.add_argument("--uma-model", default="uma-s-1p1")
    p.add_argument("--uma-task", default="oc20")
    p.add_argument("--uma-fmax", type=float, default=0.05)
    p.add_argument("--uma-max-steps", type=int, default=1000)
    p.add_argument("--uma-fire-dt", type=float, default=0.02)
    p.add_argument("--atom-budget", type=int, default=4000)
    p.add_argument("--save-every-chunks", type=int, default=20)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    from fast_dynamics import UMAWrapper, prepare_batch_for_dynamics
    from nvalchemi.data import AtomicData as NVAtomicData
    from nvalchemi.data import Batch as NVBatch
    from nvalchemi.dynamics import FIRE, ConvergenceHook

    shard_path = Path(args.shards_dir) / f"e_sys_shard{args.shard_idx}.pkl"
    with open(shard_path, "rb") as f:
        results = pickle.load(f)
    nonconv = {s for s, r in results.items() if not r.get("converged")}
    print(f"[shard{args.shard_idx}] {len(results)} records, "
          f"{len(nonconv)} non-converged", flush=True)
    if not nonconv:
        print(f"[shard{args.shard_idx}] nothing to do", flush=True)
        return

    # rebuild the exact (lmdb_id, raw_idx) work list this shard owns so each
    # non-converged sid can be matched back to its dataset metadata.
    work_keys = []
    for lid, path in enumerate(args.lmdbs):
        idxs = _clean_indices(path)
        work_keys.extend((lid, i) for i in idxs)
    mine = work_keys[args.shard_idx::args.num_shards]

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
            cell=cell.reshape(3, 3),
            tags=np.asarray(e["tags"], dtype=np.int64),
            fixed=np.asarray(e["fixed"], dtype=np.int64),
        )

    items = []
    for lid, idx in mine:
        meta = load(lid, idx)
        s = meta["sid"]
        if s not in nonconv:
            continue
        # continue from the stored partially-relaxed geometry, not the DFT one
        meta["pos"] = np.asarray(results[s]["pos_relaxed"], dtype=np.float64)
        meta["n_atoms"] = int(len(meta["numbers"]))
        meta["prev_steps"] = int(results[s].get("n_steps", 0))
        items.append(meta)
    print(f"[shard{args.shard_idx}] re-relaxing {len(items)} systems "
          f"(max_steps={args.uma_max_steps}, continuing from stored geometry)",
          flush=True)
    chunks = _chunk_by_atom_budget(items, args.atom_budget)

    uma = UMAWrapper.from_checkpoint(args.uma_model, task_name=args.uma_task,
                                     device=device)

    t0 = time.time()
    done = 0
    newconv = 0
    pbar = tqdm(chunks, desc=f"[shard{args.shard_idx}] reconverge",
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
            conv = bool(fmax_i <= args.uma_fmax)
            results[it["sid"]] = dict(
                e_total=float(energies[i]),
                converged=conv,
                fmax=fmax_i,
                n_steps=it["prev_steps"] + int(step_counter.n_steps),
                n_atoms=it["n_atoms"],
                pos_relaxed=positions_out[s:e_].cpu().numpy().astype(np.float32),
            )
            done += 1
            newconv += int(conv)
        pbar.set_postfix(done=done, newly_converged=newconv,
                         rate=f"{done/(time.time()-t0):.2f}/s")
        del nvbatch, forces, positions_out, fire
        torch.cuda.empty_cache()
        if (ci + 1) % args.save_every_chunks == 0:
            with open(shard_path, "wb") as f:
                pickle.dump(results, f)

    with open(shard_path, "wb") as f:
        pickle.dump(results, f)
    el = time.time() - t0
    still = sum(1 for r in results.values() if not r.get("converged"))
    print(f"[shard{args.shard_idx}] DONE: re-relaxed {done}, newly converged "
          f"{newconv}, still non-converged {still} in {el:.0f}s -> {shard_path}",
          flush=True)


if __name__ == "__main__":
    main()
