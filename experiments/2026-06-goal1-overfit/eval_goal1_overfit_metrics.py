#!/usr/bin/env python
"""Goal-1 overfit metrics on a fixed subset LMDB.

Metrics:
  * StructureMatcher/RMSD between flow xhat1 and target pos_relaxed.
  * UMA single-point energy and force norm for xhat1 and target.
  * Batched LBFGS steps/convergence from xhat1 and from target.
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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lmdb
import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Lattice, Structure
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO = Path(os.environ.get("ADSGEN_ROOT", "/home1/irteam/AdsorbGen")).resolve()
GEOOPT = REPO / "geoopt"
for path in (REPO, GEOOPT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement  # noqa: E402
from adsorbgen.evaluation.energy import UMAEnergy, UMAForce  # noqa: E402
from adsorbgen.flow import euler_sample  # noqa: E402
from adsorbgen.replay.eval import _model_cfg, _runtime_movable_mask  # noqa: E402
from geoopt import load_model_from_ckpt, load_uma, run_optimizer  # noqa: E402


def lmdb_len(env: lmdb.Environment) -> int:
    with env.begin() as txn:
        raw = txn.get(b"length")
        return int(pickle.loads(raw)) if raw is not None else int(txn.stat()["entries"])


def read_entry(env: lmdb.Environment, idx: int) -> dict[str, Any]:
    with env.begin() as txn:
        raw = txn.get(str(int(idx)).encode("ascii"))
    if raw is None:
        raise KeyError(idx)
    return pickle.loads(raw)


def stats(vals: list[float]) -> dict[str, float | int | None]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def to_structure(numbers, positions, cell, mask) -> Structure:
    nums = np.asarray(numbers, dtype=int)[mask]
    pos = np.asarray(positions, dtype=float)[mask]
    species = [Element.from_Z(int(z)) for z in nums.tolist()]
    return Structure(Lattice(np.asarray(cell, dtype=float)), species, pos, coords_are_cartesian=True, to_unit_cell=True)


def ordered_pbc_rms(pred, ref, cell, mask) -> float | None:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return None
    lattice = Lattice(np.asarray(cell, dtype=float))
    fp = lattice.get_fractional_coords(np.asarray(pred, dtype=float)[mask])
    fr = lattice.get_fractional_coords(np.asarray(ref, dtype=float)[mask])
    df = fp - fr
    df = df - np.round(df)
    dc = lattice.get_cartesian_coords(df)
    return float(np.sqrt(np.mean(np.sum(dc * dc, axis=1))))


def scope_mask(tags: np.ndarray, pad: np.ndarray, scope: str) -> np.ndarray:
    if scope == "all":
        return pad.astype(bool)
    if scope == "movable":
        return (tags != 0) & pad
    if scope == "ads":
        return (tags == 2) & pad
    raise ValueError(scope)


def score_structure(sm, numbers, pred, ref, cell, tags, pad, scope: str) -> dict[str, Any]:
    mask = scope_mask(tags, pad, scope)
    if int(mask.sum()) < 1:
        return {"scope": scope, "match": False, "sm_rms": None, "sm_max_dist": None, "ordered_pbc_rms_A": None, "n_atoms": 0}
    out = {
        "scope": scope,
        "match": False,
        "sm_rms": None,
        "sm_max_dist": None,
        "ordered_pbc_rms_A": ordered_pbc_rms(pred, ref, cell, mask),
        "n_atoms": int(mask.sum()),
    }
    try:
        sp = to_structure(numbers, pred, cell, mask)
        sr = to_structure(numbers, ref, cell, mask)
        out["match"] = bool(sm.fit(sp, sr))
        rms_pair = sm.get_rms_dist(sp, sr) if out["match"] else None
        if rms_pair is not None:
            out["sm_rms"] = float(rms_pair[0])
            out["sm_max_dist"] = float(rms_pair[1])
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def force_norm_rows(force: torch.Tensor, pad: torch.Tensor, tags: torch.Tensor) -> list[dict[str, float | None]]:
    out = []
    f = force.detach().float()
    norms = f.norm(dim=-1)
    for i in range(f.shape[0]):
        pad_i = pad[i].bool()
        tags_i = tags[i]
        rec = {}
        for scope in ("all", "movable", "ads"):
            if scope == "all":
                mask = pad_i
            elif scope == "movable":
                mask = pad_i & (tags_i != 0)
            else:
                mask = pad_i & (tags_i == 2)
            vals = norms[i][mask]
            rec[f"force_{scope}_max"] = float(vals.max().item()) if vals.numel() else None
            rec[f"force_{scope}_mean"] = float(vals.mean().item()) if vals.numel() else None
        out.append(rec)
    return out


class RepeatedPlacementDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        lmdb_path: str,
        num_placements: int,
        seed: int,
        use_ads_ref: bool,
        prior_mode: str,
        num_shards: int = 1,
        shard_idx: int = 0,
    ):
        self.base = PlacementPriorDataset(
            lmdb_path,
            max_samples=None,
            training_aug=False,
            unique_by_system_key=False,
            prior_mode=prior_mode,
            provide_ads_ref_pos=use_ads_ref,
            skip_anomaly=False,
            on_failure="raise",
        )
        self.num_placements = int(num_placements)
        self.seed = int(seed)
        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)
        self.n_base = lmdb_len(self.env)
        self.n_total = self.n_base * self.num_placements
        self.num_shards = int(num_shards)
        self.shard_idx = int(shard_idx)
        if self.num_shards < 1:
            raise ValueError("--num-shards must be >= 1")
        if not (0 <= self.shard_idx < self.num_shards):
            raise ValueError("--shard-idx must satisfy 0 <= shard_idx < num_shards")
        self.indices = list(range(self.shard_idx, self.n_total, self.num_shards))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        global_i = int(self.indices[int(idx)])
        raw_idx = global_i // self.num_placements
        placement_i = global_i % self.num_placements
        seed = (self.seed + raw_idx * 1009 + placement_i * 9176) & 0xFFFF_FFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sample = self.base[raw_idx]
        sample["_global_i"] = global_i
        sample["_raw_idx"] = raw_idx
        sample["_placement_i"] = placement_i
        entry = read_entry(self.env, raw_idx)
        sample["_mlip_e_total"] = float(entry["mlip_e_total"])
        sample["_mlip_fmax"] = float(entry.get("mlip_fmax", float("nan")))
        sample["_goal1_system_key"] = str(entry.get("goal1_system_key", entry.get("system_key", f"sid:{entry.get('sid', raw_idx)}")))
        return sample


def collate_with_meta(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_displacement(samples)
    for key in ("_global_i", "_raw_idx", "_placement_i", "_mlip_e_total", "_mlip_fmax", "_goal1_system_key"):
        batch[key] = [s[key] for s in samples]
    return batch


@torch.no_grad()
def build_predictions(args, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model, flow_cfg = load_model_from_ckpt(Path(args.ckpt), device)
    cfg = _model_cfg(model)
    use_ads_ref = bool(getattr(cfg, "use_ads_ref_pos", False))
    langevin_force_model = None
    if bool(getattr(cfg, "use_langevin_param", False)):
        if str(getattr(cfg, "langevin_eval_on", "x_t")) != "x_t":
            raise ValueError("Only langevin_eval_on='x_t' is implemented")
        langevin_force_model = UMAForce(model_name=args.langevin_uma_model, task_name=args.langevin_uma_task, device=str(device))

    ds = RepeatedPlacementDataset(
        args.lmdb,
        args.num_placements,
        args.seed,
        use_ads_ref,
        args.prior_mode,
        num_shards=args.num_shards,
        shard_idx=args.shard_idx,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_with_meta, pin_memory=True)
    energy = UMAEnergy(model_name=args.uma_model, task_name=args.uma_task, device=str(device), normalize_per_atom=False)
    force_model = UMAForce(model_name=args.uma_model, task_name=args.uma_task, device=str(device))
    sm = StructureMatcher(
        ltol=args.ltol,
        stol=args.stol,
        angle_tol=args.angle_tol,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
    )

    rows: list[dict[str, Any]] = []
    pred_jobs: list[dict[str, Any]] = []
    target_jobs: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="flow+sp", dynamic_ncols=True):
        global_ids = [int(x) for x in batch["_global_i"]]
        raw_idx = [int(x) for x in batch["_raw_idx"]]
        placement_i = [int(x) for x in batch["_placement_i"]]
        mlip_e_total = [float(x) for x in batch["_mlip_e_total"]]
        mlip_fmax = [float(x) for x in batch["_mlip_fmax"]]
        goal1_key = [str(x) for x in batch["_goal1_system_key"]]
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        movable = _runtime_movable_mask(model, batch)

        def fwd(x_t, t):
            extra = {}
            if use_ads_ref:
                extra["ads_ref_pos"] = batch["ads_ref_pos"]
            if langevin_force_model is not None:
                extra["mlip_force"] = langevin_force_model(
                    x_t.detach(),
                    batch["cell"],
                    batch["atomic_numbers"],
                    batch["pad_mask"],
                )
                extra["langevin_prediction_type"] = flow_cfg.prediction_type
            return model(
                pos=batch["pos"],
                x_t=x_t,
                t=t,
                atomic_numbers=batch["atomic_numbers"],
                tags=batch["tags"],
                movable_mask=movable,
                pad_mask=batch["pad_mask"],
                cell=batch["cell"],
                **extra,
            )

        pred = euler_sample(
            fwd,
            batch["pos"],
            movable,
            batch["pad_mask"],
            flow_cfg,
            num_steps=int(args.flow_steps),
            use_sde=False,
            refine_final=False,
        )
        e_pred = energy(pred, batch["cell"], batch["atomic_numbers"], batch["pad_mask"]).detach().float().cpu().numpy()
        e_target = energy(batch["pos_relaxed"], batch["cell"], batch["atomic_numbers"], batch["pad_mask"]).detach().float().cpu().numpy()
        f_pred = force_model(pred, batch["cell"], batch["atomic_numbers"], batch["pad_mask"])
        f_target = force_model(batch["pos_relaxed"], batch["cell"], batch["atomic_numbers"], batch["pad_mask"])
        pred_force_rows = force_norm_rows(f_pred, batch["pad_mask"], batch["tags"])
        target_force_rows = force_norm_rows(f_target, batch["pad_mask"], batch["tags"])

        pred_np = pred.detach().cpu().numpy()
        target_np = batch["pos_relaxed"].detach().cpu().numpy()
        nums_np = batch["atomic_numbers"].detach().cpu().numpy()
        tags_np = batch["tags"].detach().cpu().numpy()
        fixed_np = batch["fixed"].detach().cpu().numpy()
        pad_np = batch["pad_mask"].detach().cpu().numpy().astype(bool)
        cell_np = batch["cell"].detach().cpu().numpy()
        sid_np = batch["sid"].detach().cpu().numpy()
        ads_np = batch["ads_id"].detach().cpu().numpy()

        for i in range(pred_np.shape[0]):
            n = int(pad_np[i].sum())
            rec = {
                "global_i": int(global_ids[i]),
                "raw_idx": int(raw_idx[i]),
                "placement_i": int(placement_i[i]),
                "system_key": goal1_key[i],
                "sid": int(sid_np[i]),
                "ads_id": int(ads_np[i]),
                "n_atoms": int(n),
                "mlip_e_total_ref": float(mlip_e_total[i]),
                "mlip_fmax_ref": float(mlip_fmax[i]),
                "pred_E_sys": float(e_pred[i]),
                "target_sp_E_sys": float(e_target[i]),
                "pred_delta_E_vs_mlip_ref": float(e_pred[i] - mlip_e_total[i]),
                "target_sp_delta_E_vs_mlip_ref": float(e_target[i] - mlip_e_total[i]),
                "pred_delta_E_vs_target_sp": float(e_pred[i] - e_target[i]),
                "structure": [],
            }
            rec.update({f"pred_{k}": v for k, v in pred_force_rows[i].items()})
            rec.update({f"target_{k}": v for k, v in target_force_rows[i].items()})
            for scope in ("all", "movable", "ads"):
                rec["structure"].append(
                    score_structure(
                        sm,
                        nums_np[i],
                        pred_np[i],
                        target_np[i],
                        cell_np[i],
                        tags_np[i],
                        pad_np[i],
                        scope,
                    )
                )
            rows.append(rec)
            base = {
                "global_i": int(global_ids[i]),
                "raw_idx": int(raw_idx[i]),
                "placement_i": int(placement_i[i]),
                "system_key": goal1_key[i],
                "sid": int(sid_np[i]),
                "ads_id": int(ads_np[i]),
                "relax_input": {
                    "numbers": nums_np[i, :n].astype(np.int64),
                    "tags": tags_np[i, :n].astype(np.int64),
                    "fixed": fixed_np[i, :n].astype(np.int64),
                    "cell": cell_np[i].astype(np.float32),
                    "pos_pred": None,
                },
            }
            pred_job = dict(base)
            pred_job["relax_input"] = dict(base["relax_input"])
            pred_job["relax_input"]["pos_pred"] = pred_np[i, :n].astype(np.float64)
            target_job = dict(base)
            target_job["relax_input"] = dict(base["relax_input"])
            target_job["relax_input"]["pos_pred"] = target_np[i, :n].astype(np.float64)
            pred_jobs.append(pred_job)
            target_jobs.append(target_job)
    return rows, pred_jobs, target_jobs


def relax(args, jobs: list[dict[str, Any]], device: torch.device, label: str) -> list[dict[str, Any]]:
    if not jobs:
        return []
    t0 = time.time()
    uma = load_uma(
        str(args.geoopt_uma_model),
        str(args.geoopt_uma_task),
        device,
        inference_settings=str(args.geoopt_inference_settings),
        internal_graph_version=int(args.geoopt_internal_graph_version) if int(args.geoopt_internal_graph_version) > 0 else None,
        execution_mode=str(args.geoopt_execution_mode) if args.geoopt_execution_mode else None,
        compile_model=bool(args.geoopt_compile) if args.geoopt_compile else None,
    )
    opt_args = SimpleNamespace(
        fmax=float(args.fmax),
        max_steps=int(args.max_steps),
        max_atoms=int(args.max_atoms),
        maxstep=float(args.maxstep),
        lbfgs_memory=int(args.lbfgs_memory),
        lbfgs_damping=float(args.lbfgs_damping),
        lbfgs_alpha=float(args.lbfgs_alpha),
        lbfgs_check_interval=int(args.lbfgs_check_interval),
        lbfgs_streaming=bool(args.lbfgs_streaming),
        lbfgs_stream_sort=bool(args.lbfgs_stream_sort),
        lbfgs_keep_survivors_on_gpu=bool(args.lbfgs_keep_survivors_on_gpu),
        lbfgs_history_dtype=str(args.lbfgs_history_dtype),
        lbfgs_position_dtype=str(args.lbfgs_position_dtype),
        lbfgs_curvature_guard=str(args.lbfgs_curvature_guard),
        lbfgs_gpu_history_guard=bool(args.lbfgs_gpu_history_guard),
    )
    results = run_optimizer(jobs, uma, opt_args, device, "lbfgs", serial=False)
    elapsed = time.time() - t0
    by_id = {int(r["global_i"]): r for r in results}
    out = []
    for job in jobs:
        rec = dict(by_id[int(job["global_i"])])
        rec.pop("pos_relaxed", None)
        rec["source"] = label
        rec["raw_idx"] = int(job["raw_idx"])
        rec["placement_i"] = int(job["placement_i"])
        rec["system_key"] = str(job["system_key"])
        rec["sid"] = int(job["sid"])
        rec["ads_id"] = int(job["ads_id"])
        out.append(rec)
    print(json.dumps({"relax_source": label, "jobs": len(jobs), "elapsed_sec": elapsed}, sort_keys=True), flush=True)
    return out


def summarize(rows: list[dict[str, Any]], pred_relax: list[dict[str, Any]], target_relax: list[dict[str, Any]]) -> dict[str, Any]:
    by_scope: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for score in row["structure"]:
            by_scope.setdefault(score["scope"], []).append(score)
    structure_summary = {}
    for scope, vals in by_scope.items():
        structure_summary[scope] = {
            "n": len(vals),
            "match_rate": sum(1 for v in vals if v.get("match")) / max(len(vals), 1),
            "ordered_pbc_rms_A": stats([v.get("ordered_pbc_rms_A") for v in vals]),
            "sm_rms_matched": stats([v.get("sm_rms") for v in vals if v.get("match")]),
            "errors": sum(1 for v in vals if v.get("error")),
        }

    def relax_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
        conv = [v for v in vals if v.get("converged")]
        return {
            "n": len(vals),
            "converged": len(conv),
            "converged_rate": len(conv) / max(len(vals), 1),
            "steps_all": stats([v.get("n_steps") for v in vals]),
            "steps_converged": stats([v.get("n_steps") for v in conv]),
            "fmax_final": stats([v.get("fmax") for v in vals]),
            "E_sys_final": stats([v.get("E_sys") for v in vals]),
        }

    return {
        "n": len(rows),
        "structure": structure_summary,
        "energy": {
            "pred_delta_E_vs_mlip_ref": stats([r["pred_delta_E_vs_mlip_ref"] for r in rows]),
            "target_sp_delta_E_vs_mlip_ref": stats([r["target_sp_delta_E_vs_mlip_ref"] for r in rows]),
            "pred_delta_E_vs_target_sp": stats([r["pred_delta_E_vs_target_sp"] for r in rows]),
        },
        "force": {
            "pred_movable_fmax": stats([r["pred_force_movable_max"] for r in rows]),
            "target_movable_fmax": stats([r["target_force_movable_max"] for r in rows]),
            "pred_ads_fmax": stats([r["pred_force_ads_max"] for r in rows]),
            "target_ads_fmax": stats([r["target_force_ads_max"] for r in rows]),
        },
        "relax_pred": relax_summary(pred_relax),
        "relax_target": relax_summary(target_relax),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-placements", type=int, default=3)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--langevin-uma-model", default="uma-s-1p2")
    ap.add_argument("--langevin-uma-task", default="oc20")
    ap.add_argument("--geoopt-uma-model", default="uma-s-1p1")
    ap.add_argument("--geoopt-uma-task", default="oc20")
    ap.add_argument("--geoopt-inference-settings", default="default")
    ap.add_argument("--geoopt-internal-graph-version", type=int, default=0)
    ap.add_argument("--geoopt-execution-mode", default="")
    ap.add_argument("--geoopt-compile", action="store_true")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-atoms", type=int, default=4096)
    ap.add_argument("--maxstep", type=float, default=0.04)
    ap.add_argument("--lbfgs-memory", type=int, default=50)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    ap.add_argument("--lbfgs-check-interval", type=int, default=20)
    ap.add_argument("--lbfgs-streaming", action="store_true")
    ap.add_argument("--lbfgs-stream-sort", action="store_true")
    ap.add_argument("--lbfgs-keep-survivors-on-gpu", action="store_true")
    ap.add_argument("--lbfgs-history-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-position-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-curvature-guard", choices=["abs", "positive", "ase"], default="abs")
    ap.add_argument("--lbfgs-gpu-history-guard", action="store_true")
    ap.add_argument("--ltol", type=float, default=0.3)
    ap.add_argument("--stol", type=float, default=0.5)
    ap.add_argument("--angle-tol", type=float, default=10.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")

    rows, pred_jobs, target_jobs = build_predictions(args, device)
    pred_relax = relax(args, pred_jobs, device, "pred")
    target_relax = relax(args, target_jobs, device, "target")
    payload = {
        "settings": vars(args),
        "summary": summarize(rows, pred_relax, target_relax),
        "rows": rows,
        "pred_relax": pred_relax,
        "target_relax": target_relax,
    }
    tmp = out_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, out_dir / "summary.json")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
