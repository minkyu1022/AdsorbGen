#!/usr/bin/env python
"""Relax every OC20-Dense adslab config with UMA oc20 and save E_sys + geometry."""

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

from adsorbgen.replay.eval import _chunk_by_atom_budget  # noqa: E402
from adsorbgen.replay.viz import FixedAtomsHook  # noqa: E402


class StepCounterHook:
    def __init__(self):
        from nvalchemi.dynamics import DynamicsStage

        self.frequency = 1
        self.stage = DynamicsStage.AFTER_STEP
        self.n_steps = 0

    def __call__(self, ctx, stage) -> None:  # noqa: ARG002
        self.n_steps += 1


def _load_entry(txn, idx: int) -> dict:
    e = pickle.loads(txn.get(str(idx).encode()))
    cell = np.asarray(e["cell"], dtype=np.float32)
    if cell.ndim == 3:
        cell = cell[0]
    return {
        "idx": int(idx),
        "system_key": str(e["system_key"]),
        "config_key": str(e.get("config_key", idx)),
        "ads_id": int(e["ads_id"]),
        "numbers": np.asarray(e["atomic_numbers"], dtype=np.int64),
        "pos": np.asarray(e["pos_relaxed"], dtype=np.float64),
        "cell": cell.reshape(3, 3),
        "tags": np.asarray(e["tags"], dtype=np.int64),
        "fixed": np.asarray(e["fixed"], dtype=np.int64),
        "n_atoms": int(len(e["atomic_numbers"])),
        "y_relaxed_dft": float(e.get("y_relaxed", np.nan)),
        "delta_e_dft": float(e.get("delta_e", np.nan)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="/home/irteam/data/processed/oc20dense.lmdb")
    ap.add_argument("--out-dir", default="/home/irteam/data/replay/oc20dense_mlip_relax_shards")
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--uma-fmax", type=float, default=0.05)
    ap.add_argument("--uma-max-steps", type=int, default=300)
    ap.add_argument("--uma-fire-dt", type=float, default=0.02)
    ap.add_argument("--atom-budget", type=int, default=4000)
    ap.add_argument("--save-every-chunks", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    from fast_dynamics import UMAWrapper, prepare_batch_for_dynamics
    from nvalchemi.data import AtomicData as NVAtomicData
    from nvalchemi.data import Batch as NVBatch
    from nvalchemi.dynamics import FIRE, ConvergenceHook

    out_path = Path(args.out_dir) / f"oc20dense_mlip_relax_shard{args.shard_idx}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict] = {}
    if args.resume and out_path.exists():
        with out_path.open("rb") as f:
            results = pickle.load(f)
        print(f"[dense shard{args.shard_idx}] resume loaded {len(results)} records", flush=True)

    env = lmdb.open(args.lmdb, subdir=False, readonly=True, lock=False, readahead=False)
    txn = env.begin()
    n_total = int(pickle.loads(txn.get(b"length")))
    indices = list(range(n_total))[args.shard_idx::args.num_shards]
    if results:
        indices = [i for i in indices if i not in results]
    print(
        f"[dense shard{args.shard_idx}] assigned {len(indices)} / {n_total}; "
        f"UMA={args.uma_model} task={args.uma_task} max_steps={args.uma_max_steps}",
        flush=True,
    )

    uma = UMAWrapper.from_checkpoint(args.uma_model, task_name=args.uma_task, device=device)
    items = [_load_entry(txn, i) for i in indices]
    chunks = _chunk_by_atom_budget(items, args.atom_budget)

    t0 = time.time()
    pbar = tqdm(chunks, desc=f"[dense shard{args.shard_idx}] UMA relax", unit="chunk", dynamic_ncols=True)
    for ci, chunk in enumerate(pbar):
        finite = [it for it in chunk if np.isfinite(it["pos"]).all()]
        if not finite:
            continue
        data_list = [
            NVAtomicData(
                positions=torch.as_tensor(it["pos"], dtype=torch.float32, device=device),
                atomic_numbers=torch.as_tensor(it["numbers"], dtype=torch.long, device=device),
                cell=torch.as_tensor(it["cell"], dtype=torch.float32, device=device).reshape(1, 3, 3),
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
        fixed_mask = torch.as_tensor(np.concatenate(fixed_parts), dtype=torch.bool, device=device)

        fire = FIRE(
            uma,
            dt=args.uma_fire_dt,
            n_steps=args.uma_max_steps,
            convergence_hook=ConvergenceHook.from_fmax(args.uma_fmax),
        )
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
            s, e = ptr[i], ptr[i + 1]
            fmax_i = float(forces[s:e].norm(dim=-1).max().item())
            results[int(it["idx"])] = {
                "idx": int(it["idx"]),
                "system_key": it["system_key"],
                "config_key": it["config_key"],
                "ads_id": int(it["ads_id"]),
                "e_total": float(energies[i]),
                "converged": bool(fmax_i <= args.uma_fmax),
                "fmax": fmax_i,
                "n_steps": int(step_counter.n_steps),
                "n_atoms": int(it["n_atoms"]),
                "pos_relaxed": positions_out[s:e].cpu().numpy().astype(np.float32),
                "y_relaxed_dft": float(it["y_relaxed_dft"]),
                "delta_e_dft": float(it["delta_e_dft"]),
            }

        n_conv = sum(1 for r in results.values() if r["converged"])
        elapsed = max(time.time() - t0, 1e-6)
        pbar.set_postfix(records=len(results), conv=n_conv, rate=f"{len(results)/elapsed:.2f}/s")
        del nvbatch, fire, forces, positions_out
        torch.cuda.empty_cache()

        if (ci + 1) % args.save_every_chunks == 0:
            with out_path.open("wb") as f:
                pickle.dump(results, f)

    with out_path.open("wb") as f:
        pickle.dump(results, f)
    elapsed = time.time() - t0
    print(
        f"[dense shard{args.shard_idx}] DONE {len(results)} records in {elapsed:.0f}s "
        f"({len(results)/max(elapsed, 1e-6):.2f}/s) -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
