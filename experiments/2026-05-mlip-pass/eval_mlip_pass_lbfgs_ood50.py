#!/usr/bin/env python
"""AdsorbSample-style MLIP Pass@k sanity check on OC20-Dense OOD-50.

This script evaluates one shard:
  model inference -> UMA/OC20 ASE L-BFGS relaxation -> strict validity ->
  MLIP Pass@k against the 100-system OC20-Dense cover reference minima.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from tqdm.auto import tqdm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement  # noqa: E402
from adsorbgen.evaluation.metrics import load_pristine_context, _score_record_anomaly  # noqa: E402
from adsorbgen.replay.eval import _model_cfg, _runtime_movable_mask  # noqa: E402
from adsorbgen.flow import FlowConfig, euler_sample  # noqa: E402
from adsorbgen.models.dit import DiTDenoiserConfig  # noqa: E402
from adsorbgen.models.factory import build_model  # noqa: E402
from adsorbgen.models.dit_v2 import DiTDenoiserV2Config  # noqa: E402


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    torch.serialization.add_safe_globals(
        [DiTDenoiserConfig, DiTDenoiserV2Config, FlowConfig]
    )
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    model = build_model(hp["model_cfg"])
    sd = ck["state_dict"]
    stripped = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(stripped, strict=False)
    model.adsorbgen_movable_mode = str(hp.get("movable_mode", "surface_ads"))
    model.to(device).eval()
    return model, hp["flow_cfg"]


def load_selected_systems(args) -> list[str]:
    split = json.load(open(args.split_membership))
    rows = split["subset100_rows"]
    ood = sorted(r["system_key"] for r in rows if str(r["class"]).startswith("val_ood"))
    if len(ood) < args.num_systems:
        raise ValueError(f"only {len(ood)} OOD systems available, need {args.num_systems}")
    rng = np.random.default_rng(args.seed)
    return sorted(rng.choice(ood, size=args.num_systems, replace=False).tolist())


def load_reference_by_system(cover_dir: Path) -> dict[str, dict]:
    gm = json.load(open(cover_dir / "gt_results" / "global_minima.json"))
    by_system = {}
    for rec in gm.values():
        system_key = str(rec["system_id"])
        by_system[system_key] = {
            "system_key": system_key,
            "E_ads_ref": float(rec["relaxed_e_ads"]),
            "E_sys_ref": float(rec["e_sys_relaxed"]),
            "E_slab": float(rec["e_slab_relaxed"]),
            "E_adsorbate": float(rec["e_adsorbate"]),
            "config_id_ref": rec.get("config_id"),
            "subset_idx": int(rec.get("subset_idx", -1)),
        }
    return by_system


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


def atoms_from_prediction(p: dict) -> Atoms:
    atoms = Atoms(
        numbers=np.asarray(p["numbers"], dtype=int),
        positions=np.asarray(p["pos_pred"], dtype=float),
        cell=np.asarray(p["cell"], dtype=float),
        pbc=True,
        tags=np.asarray(p["tags"], dtype=int).tolist(),
    )
    fixed = np.asarray(p["fixed"], dtype=bool)
    if not fixed.any():
        fixed = np.asarray(p["tags"], dtype=int) == 0
    if fixed.any():
        atoms.set_constraint(FixAtoms(indices=np.where(fixed)[0].tolist()))
    return atoms


def relax_lbfgs_one(p: dict, calc, args) -> dict:
    atoms = atoms_from_prediction(p)
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
        e_sys = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        fmax = float(np.max(np.linalg.norm(forces, axis=1)))
        relaxed_pos = atoms.get_positions().astype(np.float32)
        err = None
    except Exception as exc:
        converged = False
        e_sys = float("nan")
        fmax = float("nan")
        relaxed_pos = np.asarray(p["pos_pred"], dtype=np.float32)
        err = repr(exc)
    return {
        "converged": converged,
        "E_sys": e_sys,
        "fmax": fmax,
        "n_steps": int(getattr(opt, "nsteps", 0)),
        "pos_relaxed": relaxed_pos,
        "error": err,
    }


def pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    if k > n:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def summarize(rows: list[dict], systems: list[str], k_values=(1, 2, 5, 10)) -> dict:
    by_sys = defaultdict(list)
    for r in rows:
        by_sys[str(r["system_key"])].append(r)

    per_system = {}
    for sk in systems:
        rs = by_sys.get(sk, [])
        n = len(rs)
        c = sum(1 for r in rs if r.get("success"))
        per_system[sk] = {
            "n": n,
            "c_success": c,
            "n_valid": sum(1 for r in rs if r.get("valid")),
            "n_converged": sum(1 for r in rs if r.get("converged")),
            **{f"pass@{k}": pass_at_k(n, c, k) for k in k_values},
        }

    denom = max(len(systems), 1)
    total = max(len(rows), 1)
    return {
        "systems": len(systems),
        "candidates": len(rows),
        "valid_rate": sum(1 for r in rows if r.get("valid")) / total,
        "converged_rate": sum(1 for r in rows if r.get("converged")) / total,
        "success_sample_rate": sum(1 for r in rows if r.get("success")) / total,
        **{
            f"mlip_pass@{k}": sum(v[f"pass@{k}"] for v in per_system.values()) / denom
            for k in k_values
        },
        "per_system": per_system,
    }


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/irteam/runs/H200_ads_pair_dist_loss/ckpt_epochepoch=099.ckpt")
    ap.add_argument("--lmdb", default="/home/irteam/data/processed/oc20dense.lmdb")
    ap.add_argument("--cover-dir", default="/home/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover")
    ap.add_argument("--split-membership", default="/home/irteam/data/replay/oc20dense_oc20_split_membership.json")
    ap.add_argument("--out-dir", default="/home/irteam/data/replay/mlip_pass_lbfgs_ood50")
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260520)
    ap.add_argument("--num-systems", type=int, default=50)
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--flow-batch-size", type=int, default=32)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--use-sde", action="store_true")
    ap.add_argument("--refine-final", action="store_true")
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

    selected_path = out_dir / "selected_ood50_systems.json"
    if args.shard_idx == 0:
        selected_path.write_text(json.dumps({
            "seed": args.seed,
            "num_systems": args.num_systems,
            "systems": selected_systems,
            "reference_source": str(Path(args.cover_dir) / "gt_results" / "global_minima.json"),
        }, indent=2, sort_keys=True))

    raw_idx_by_system = map_system_to_raw_idx(args.lmdb, set(selected_systems))
    all_tasks = []
    for sys_i, sk in enumerate(selected_systems):
        for sample_i in range(args.num_samples):
            global_i = sys_i * args.num_samples + sample_i
            if global_i % args.num_shards == args.shard_idx:
                all_tasks.append((global_i, sk, sample_i, raw_idx_by_system[sk]))

    print(
        f"[shard{args.shard_idx}] systems={len(selected_systems)} "
        f"tasks={len(all_tasks)}/{len(selected_systems) * args.num_samples}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this evaluation shard")
    model, flow_cfg = load_model_from_ckpt(Path(args.ckpt), device)
    use_ads_ref = bool(getattr(_model_cfg(model), "use_ads_ref_pos", False))

    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    predict_unit = pretrained_mlip.get_predict_unit(args.uma_model, device=str(device))
    calc = FAIRChemCalculator(predict_unit, task_name=args.uma_task)
    load_pristine_context(Path(args.pristine_slabs), Path(args.pristine_index))

    placement_ds = PlacementPriorDataset(
        args.lmdb,
        prior_mode=args.prior_mode,
        max_samples=None,
        provide_ads_ref_pos=use_ads_ref,
        skip_anomaly=False,
    )

    rows = []
    t0 = time.time()
    for start in tqdm(
        range(0, len(all_tasks), args.flow_batch_size),
        desc=f"[shard{args.shard_idx}] flow+lbfgs",
        unit="batch",
        dynamic_ncols=True,
    ):
        chunk_tasks = all_tasks[start:start + args.flow_batch_size]
        samples = []
        metas = []
        for global_i, sk, sample_i, raw_idx in chunk_tasks:
            np.random.seed((args.seed + global_i) & 0xFFFF_FFFF)
            random.seed((args.seed + global_i) & 0xFFFF_FFFF)
            sample = placement_ds[int(raw_idx)]
            samples.append(sample)
            metas.append((global_i, sk, sample_i, raw_idx))

        batch = collate_displacement(samples)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        movable = _runtime_movable_mask(model, batch)

        def fwd(x_t, t, _b=batch):
            extra = {}
            if use_ads_ref:
                extra["ads_ref_pos"] = _b["ads_ref_pos"]
            return model(
                pos=_b["pos"],
                x_t=x_t,
                t=t,
                atomic_numbers=_b["atomic_numbers"],
                tags=_b["tags"],
                movable_mask=movable,
                pad_mask=_b["pad_mask"],
                cell=_b["cell"],
                **extra,
            )

        x_out = euler_sample(
            fwd,
            batch["pos"],
            movable,
            batch["pad_mask"],
            flow_cfg,
            num_steps=args.flow_steps,
            use_sde=args.use_sde,
            refine_final=args.refine_final,
        )

        for i, (global_i, sk, sample_i, raw_idx) in enumerate(metas):
            n = int(batch["pad_mask"][i].sum().item())
            tags = batch["tags"][i, :n].detach().cpu().numpy().astype(np.int64)
            numbers = batch["atomic_numbers"][i, :n].detach().cpu().numpy().astype(np.int64)
            fixed = batch["fixed"][i, :n].detach().cpu().numpy().astype(np.int64)
            cell = batch["cell"][i].detach().cpu().numpy()
            if cell.ndim == 3:
                cell = cell[0]
            pos_ref = batch["pos"][i, :n].detach().cpu().numpy().astype(np.float64)
            pos_pred = x_out[i, :n].detach().cpu().numpy().astype(np.float64)
            pos_gt = batch["pos_relaxed"][i, :n].detach().cpu().numpy().astype(np.float64)
            ads_id = int(batch["ads_id"][i].item()) if "ads_id" in batch else int(samples[i]["ads_id"].item())

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
        "lbfgs": {
            "fmax": args.uma_fmax,
            "max_steps": args.uma_max_steps,
            "maxstep": args.lbfgs_maxstep,
            "memory": args.lbfgs_memory,
            "damping": args.lbfgs_damping,
            "alpha": args.lbfgs_alpha,
        },
        "sampling": {
            "flow_steps": args.flow_steps,
            "prior_mode": args.prior_mode,
            "use_sde": args.use_sde,
            "refine_final": args.refine_final,
        },
    })

    shard_pkl = out_dir / f"shard_{args.shard_idx}.pkl"
    shard_json = out_dir / f"shard_{args.shard_idx}.json"
    with shard_pkl.open("wb") as f:
        pickle.dump(rows, f)
    shard_json.write_text(json.dumps(shard_summary, indent=2, sort_keys=True))
    print(json.dumps(shard_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
