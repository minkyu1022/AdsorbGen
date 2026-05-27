#!/usr/bin/env python
"""Random-placement MLIP Pass@k baseline on the same OC20-Dense OOD-50 set.

This is the Baseline 2 analogue of ``eval_mlip_pass_lbfgs_ood50.py``:
random_heuristic placement is used directly as the candidate structure, then
the candidate is relaxed with UMA/OC20 ASE L-BFGS and scored with the same
strict validity filter and MLIP reference-minimum threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm.auto import tqdm

REPO = Path(os.environ.get("ADSGEN_ROOT", Path(__file__).resolve().parents[2]))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCRIPT_DIR = REPO / "experiments" / "2026-05-mlip-pass"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adsorbgen.data.dataset import PlacementPriorDataset  # noqa: E402
from adsorbgen.evaluation.metrics import load_pristine_context, _score_record_anomaly  # noqa: E402
from eval_mlip_pass_lbfgs_ood50 import (  # noqa: E402
    atoms_from_prediction,
    load_reference_by_system,
    load_selected_systems,
    pass_at_k,
    relax_lbfgs_one,
    summarize,
)


def map_system_to_raw_idx(lmdb_path: str, systems: set[str]) -> dict[str, int]:
    out = {}
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        n = int(pickle.loads(txn.get(b"length")))
        for i in range(n):
            raw = txn.get(str(i).encode())
            if raw is None:
                continue
            e = pickle.loads(raw)
            sk = str(e.get("system_key"))
            if sk in systems and sk not in out:
                out[sk] = i
                if len(out) == len(systems):
                    break
    env.close()
    missing = sorted(systems - set(out))
    if missing:
        raise KeyError(f"missing systems in {lmdb_path}: {missing[:5]}")
    return out


def _as_numpy_1d(x, dtype):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _as_numpy_pos(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _sample_int(sample: dict, key: str, default: int = -1) -> int:
    if key not in sample:
        return default
    val = sample[key]
    if isinstance(val, torch.Tensor):
        return int(val.item())
    return int(val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="/home/irteam/data-vol1/minkyu/data/processed_old/oc20dense.lmdb")
    ap.add_argument("--cover-dir", default="/home/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover")
    ap.add_argument("--split-membership", default="/home/irteam/data/replay/oc20dense_oc20_split_membership.json")
    ap.add_argument("--out-dir", default="/home/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline2_random")
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260520)
    ap.add_argument("--num-systems", type=int, default=50)
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--uma-fmax", type=float, default=0.01)
    ap.add_argument("--uma-max-steps", type=int, default=300)
    ap.add_argument("--lbfgs-maxstep", type=float, default=0.04)
    ap.add_argument("--lbfgs-memory", type=int, default=50)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    ap.add_argument("--epsilon-succ", type=float, default=0.1)
    ap.add_argument("--pristine-slabs", default="/home/irteam/results/pristine_slabs/oc20dense_uma.pkl")
    ap.add_argument("--pristine-index", default="/home/irteam/results/pristine_slabs/oc20dense.system_index.pkl")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed + args.shard_idx)
    np.random.seed(args.seed + args.shard_idx)
    torch.manual_seed(args.seed + args.shard_idx)

    selected_systems = load_selected_systems(args)
    refs = load_reference_by_system(Path(args.cover_dir))
    missing_refs = sorted(set(selected_systems) - set(refs))
    if missing_refs:
        raise KeyError(f"selected systems missing cover refs: {missing_refs[:5]}")

    if args.shard_idx == 0:
        (out_dir / "selected_ood50_systems.json").write_text(json.dumps({
            "seed": args.seed,
            "num_systems": args.num_systems,
            "systems": selected_systems,
            "reference_source": str(Path(args.cover_dir) / "gt_results" / "global_minima.json"),
            "generator": "baseline2_random_heuristic",
        }, indent=2, sort_keys=True))

    raw_idx_by_system = map_system_to_raw_idx(args.lmdb, set(selected_systems))
    all_tasks = []
    for sys_i, sk in enumerate(selected_systems):
        for sample_i in range(args.num_samples):
            global_i = sys_i * args.num_samples + sample_i
            if global_i % args.num_shards == args.shard_idx:
                all_tasks.append((global_i, sk, sample_i, raw_idx_by_system[sk]))

    print(
        f"[B2 shard{args.shard_idx}] systems={len(selected_systems)} "
        f"tasks={len(all_tasks)}/{len(selected_systems) * args.num_samples}",
        flush=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation shard")
    device = torch.device("cuda")

    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    predict_unit = pretrained_mlip.get_predict_unit(args.uma_model, device=str(device))
    calc = FAIRChemCalculator(predict_unit, task_name=args.uma_task)
    load_pristine_context(Path(args.pristine_slabs), Path(args.pristine_index))

    placement_ds = PlacementPriorDataset(
        args.lmdb,
        prior_mode=args.prior_mode,
        max_samples=None,
        provide_ads_ref_pos=False,
        skip_anomaly=False,
    )

    rows = []
    t0 = time.time()
    for global_i, sk, sample_i, raw_idx in tqdm(
        all_tasks,
        desc=f"[B2 shard{args.shard_idx}] random+lbfgs",
        unit="cand",
        dynamic_ncols=True,
    ):
        np.random.seed((args.seed + global_i) & 0xFFFF_FFFF)
        random.seed((args.seed + global_i) & 0xFFFF_FFFF)
        sample = placement_ds[int(raw_idx)]

        tags = _as_numpy_1d(sample["tags"], np.int64)
        numbers = _as_numpy_1d(sample["atomic_numbers"], np.int64)
        fixed = _as_numpy_1d(sample.get("fixed", np.zeros_like(tags)), np.int64)
        cell = _as_numpy_pos(sample["cell"])
        if cell.ndim == 3:
            cell = cell[0]
        pos_ref = _as_numpy_pos(sample["pos"])
        pos_pred = pos_ref.copy()
        pos_gt = _as_numpy_pos(sample["pos_relaxed"])
        ads_id = _sample_int(sample, "ads_id", -1)

        pred = {
            "numbers": numbers,
            "tags": tags,
            "fixed": fixed,
            "cell": cell,
            "pos_pred": pos_pred,
        }
        relaxed = relax_lbfgs_one(pred, calc, args)
        ref = refs[sk]

        status = "ok"
        valid = False
        success = False
        anomaly = None
        e_ads = float("nan")
        if not relaxed["converged"] or not np.isfinite(relaxed["E_sys"]):
            status = "uma_unconverged"
        else:
            e_ads = float(relaxed["E_sys"] - ref["E_slab"] - ref["E_adsorbate"])
            rec = {
                "sid": -1,
                "system_key": sk,
                "ads_id": ads_id,
                "pos_ref": torch.as_tensor(pos_ref, dtype=torch.float32),
                "pos_pred": torch.as_tensor(relaxed["pos_relaxed"], dtype=torch.float32),
                "pos_gt": torch.as_tensor(pos_gt, dtype=torch.float32),
                "atomic_numbers": torch.as_tensor(numbers, dtype=torch.long),
                "tags": torch.as_tensor(tags, dtype=torch.long),
                "cell": torch.as_tensor(cell, dtype=torch.float32),
            }
            ar = _score_record_anomaly(rec)
            if ar.get("valid_strict"):
                valid = True
                success = bool(e_ads - ref["E_ads_ref"] <= args.epsilon_succ)
            else:
                flags = [
                    k for k in ("overlap", "dissoc", "desorbed", "intercalated", "surf_changed")
                    if ar.get(f"has_{k}")
                ]
                anomaly = flags[0] if flags else ar.get("error") or "anomaly"
                status = anomaly

        rows.append({
            "global_i": int(global_i),
            "system_key": sk,
            "sample_i": int(sample_i),
            "raw_idx": int(raw_idx),
            "ads_id": int(ads_id),
            "E_sys": float(relaxed["E_sys"]),
            "E_ads": float(e_ads),
            "E_ads_ref": float(ref["E_ads_ref"]),
            "delta_E_ads": float(e_ads - ref["E_ads_ref"]) if np.isfinite(e_ads) else float("nan"),
            "fmax": float(relaxed["fmax"]),
            "n_steps": int(relaxed["n_steps"]),
            "converged": bool(relaxed["converged"]),
            "valid": bool(valid),
            "success": bool(success),
            "status": status,
            "anomaly": anomaly,
            "relax_error": relaxed["error"],
        })

    shard_summary = summarize(rows, selected_systems)
    shard_summary.update({
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
        "elapsed_sec": time.time() - t0,
        "generator": "baseline2_random_heuristic",
        "lbfgs": {
            "fmax": args.uma_fmax,
            "max_steps": args.uma_max_steps,
            "maxstep": args.lbfgs_maxstep,
            "memory": args.lbfgs_memory,
            "damping": args.lbfgs_damping,
            "alpha": args.lbfgs_alpha,
        },
        "sampling": {
            "prior_mode": args.prior_mode,
        },
    })

    with (out_dir / f"shard_{args.shard_idx}.pkl").open("wb") as f:
        pickle.dump(rows, f)
    (out_dir / f"shard_{args.shard_idx}.json").write_text(
        json.dumps(shard_summary, indent=2, sort_keys=True)
    )
    print(json.dumps(shard_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
