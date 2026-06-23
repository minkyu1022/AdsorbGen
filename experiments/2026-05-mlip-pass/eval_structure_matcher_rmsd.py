#!/usr/bin/env python
"""Evaluate inference structure match/RMSD for selected train and OOD systems."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Lattice, Structure

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement  # noqa: E402
from adsorbgen.evaluation.energy import UMAForce  # noqa: E402
from adsorbgen.flow import FlowConfig, euler_sample  # noqa: E402
from adsorbgen.models.dit import DiTDenoiserConfig  # noqa: E402
from adsorbgen.models.dit_v2 import DiTDenoiserV2Config  # noqa: E402
from adsorbgen.models.factory import build_model  # noqa: E402


MODELS = {
    "base": {
        "ckpt": "/home1/irteam/runs/training/ID_mlip_pairdist_only_1x_bs64_expand_20260521_213544/last.ckpt",
        "slab_source": "initial",
    },
    "x1_LP": {
        "ckpt": "/home1/irteam/runs/training/x1_LP_102M/ckpt_epochepoch=099.ckpt",
        "slab_source": "initial",
    },
    "x1_LP_bare_slab": {
        "ckpt": "/home1/irteam/runs/training/x1_LP_102M_relaxed_bare_slab_x0/last.ckpt",
        "slab_source": "pristine_relaxed",
    },
    "B1_adsorbate_only": {
        "ckpt": "/home1/irteam/runs/B1_adsorbate_only_pair_dist_loss/last.ckpt",
        "slab_source": "initial",
    },
}


def _existing(*paths: str) -> str:
    for p in paths:
        if p and Path(p).exists():
            return p
    raise FileNotFoundError(paths)


def _lmdb_len(env: lmdb.Environment) -> int:
    with env.begin() as txn:
        raw = txn.get(b"length")
        if raw is not None:
            return int(pickle.loads(raw))
        return int(txn.stat()["entries"])


def _read_entry(env: lmdb.Environment, idx: int) -> dict:
    with env.begin() as txn:
        raw = txn.get(str(int(idx)).encode("ascii"))
    if raw is None:
        raise KeyError(idx)
    return pickle.loads(raw)


def _system_key(entry: dict, idx: int) -> str:
    if "system_key" in entry:
        return str(entry["system_key"])
    sid = int(entry.get("sid", -1))
    return f"sid:{sid}" if sid >= 0 else f"idx:{idx}"


def _clean_indices(env: lmdb.Environment, n: int) -> np.ndarray:
    with env.begin() as txn:
        raw = txn.get(b"anomaly_mask")
    if raw is None:
        return np.arange(n, dtype=np.int64)
    mask = np.asarray(pickle.loads(raw), dtype=np.int8)[:n]
    return np.where(mask == 0)[0].astype(np.int64)


def _sample_train_unique(lmdb_path: str, n_select: int, seed: int) -> list[dict]:
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)
    n = _lmdb_len(env)
    candidates = _clean_indices(env, n)
    rng = np.random.default_rng(seed)
    order = rng.permutation(candidates)
    rows: list[dict] = []
    seen = set()
    for idx in order.tolist():
        entry = _read_entry(env, int(idx))
        sk = _system_key(entry, int(idx))
        if sk in seen:
            continue
        seen.add(sk)
        rows.append({"split": "train", "raw_idx": int(idx), "system_key": sk})
        if len(rows) >= n_select:
            break
    env.close()
    if len(rows) < n_select:
        raise RuntimeError(f"only selected {len(rows)} train systems")
    return rows


def _map_systems_to_indices(lmdb_path: str, systems: list[str]) -> list[dict]:
    wanted = set(map(str, systems))
    out: dict[str, int] = {}
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)
    n = _lmdb_len(env)
    for idx in range(n):
        entry = _read_entry(env, idx)
        sk = _system_key(entry, idx)
        if sk in wanted and sk not in out:
            out[sk] = idx
            if len(out) == len(wanted):
                break
    env.close()
    missing = sorted(wanted - set(out))
    if missing:
        raise KeyError(f"missing OOD systems in {lmdb_path}: {missing[:5]}")
    return [{"split": "ood", "raw_idx": int(out[sk]), "system_key": sk} for sk in systems]


def build_or_load_selection(args) -> dict:
    sel_path = Path(args.selection)
    if sel_path.exists() and not args.rebuild_selection:
        return json.loads(sel_path.read_text())

    train_rows = _sample_train_unique(args.train_lmdb, args.num_systems, args.seed)
    split = json.loads(Path(args.split_membership).read_text())
    ood_all = sorted(
        str(r["system_key"])
        for r in split["subset100_rows"]
        if str(r["class"]).startswith("val_ood")
    )
    rng = np.random.default_rng(args.seed + 17)
    ood_systems = sorted(rng.choice(ood_all, size=args.num_systems, replace=False).tolist())
    ood_rows = _map_systems_to_indices(args.ood_lmdb, ood_systems)
    payload = {
        "seed": args.seed,
        "num_systems": args.num_systems,
        "train_lmdb": args.train_lmdb,
        "ood_lmdb": args.ood_lmdb,
        "train": train_rows,
        "ood": ood_rows,
    }
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    sel_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def load_model(ckpt_path: str, device: torch.device):
    torch.serialization.add_safe_globals([DiTDenoiserConfig, DiTDenoiserV2Config, FlowConfig])
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    model = build_model(hp["model_cfg"])
    sd = ck["state_dict"]
    stripped = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(stripped, strict=False)
    model.adsorbgen_movable_mode = str(hp.get("movable_mode", "surface_ads"))
    model.to(device).eval()
    return model, hp["flow_cfg"], ck


def runtime_movable_mask(model, batch: dict) -> torch.Tensor:
    mode = str(getattr(model, "adsorbgen_movable_mode", "surface_ads"))
    tags = batch["tags"]
    pad = batch["pad_mask"].bool()
    if mode == "ads_only":
        return (tags == 2) & pad
    if mode == "surface_ads":
        return (tags != 0) & pad
    if mode == "not_fixed":
        return (~batch["fixed"].bool()) & pad
    return (tags != 0) & pad


def make_dataset(args, split_name: str, rows: list[dict], model, model_spec: dict):
    slab_source = str(model_spec["slab_source"])
    pristine_slabs = ""
    pristine_index = ""
    if slab_source == "pristine_relaxed":
        if split_name == "train":
            pristine_slabs = args.train_pristine_slabs
            pristine_index = args.train_pristine_index
        else:
            pristine_slabs = args.ood_pristine_slabs
            pristine_index = args.ood_pristine_index
    use_ads_ref = bool(getattr(model.cfg, "use_ads_ref_pos", False))
    lmdb_path = args.train_lmdb if split_name == "train" else args.ood_lmdb
    ds = PlacementPriorDataset(
        lmdb_path,
        max_samples=None,
        training_aug=False,
        unique_by_system_key=False,
        prior_mode=args.prior_mode,
        provide_ads_ref_pos=use_ads_ref,
        adsorbates_pkl=args.adsorbates_pkl,
        skip_anomaly=False,
        slab_source=slab_source,
        pristine_slabs=pristine_slabs,
        pristine_index=pristine_index,
        on_failure="raise",
    )
    ds._idx_map = np.asarray([int(r["raw_idx"]) for r in rows], dtype=np.int64)
    ds.n = len(rows)
    return ds


def seeded_samples(ds, rows: list[dict], seed: int) -> list[dict]:
    samples = []
    for local_i, row in enumerate(rows):
        s = int(seed) + int(row["raw_idx"]) * 1009 + local_i
        random.seed(s)
        np.random.seed(s % (2**32 - 1))
        torch.manual_seed(s)
        sample = ds[local_i]
        sample["_raw_idx"] = int(row["raw_idx"])
        sample["_system_key"] = str(row["system_key"])
        samples.append(sample)
    return samples


def to_structure(numbers, positions, cell, mask) -> Structure:
    nums = np.asarray(numbers, dtype=int)[mask]
    pos = np.asarray(positions, dtype=float)[mask]
    species = [Element.from_Z(int(z)) for z in nums.tolist()]
    return Structure(Lattice(np.asarray(cell, dtype=float)), species, pos, coords_are_cartesian=True, to_unit_cell=True)


def ordered_pbc_rms(numbers, pred, ref, cell, mask) -> float | None:
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


def score_structure(sm: StructureMatcher, numbers, pred, ref, cell, tags, pad, scope: str) -> dict:
    mask = scope_mask(tags, pad, scope)
    if int(mask.sum()) < 1:
        return {"scope": scope, "match": False, "sm_rms": None, "sm_max_dist": None, "ordered_pbc_rms_A": None, "n_atoms": 0}
    try:
        sp = to_structure(numbers, pred, cell, mask)
        sr = to_structure(numbers, ref, cell, mask)
        match = bool(sm.fit(sp, sr))
        rms_pair = sm.get_rms_dist(sp, sr) if match else None
        sm_rms = None if rms_pair is None else float(rms_pair[0])
        sm_max = None if rms_pair is None else float(rms_pair[1])
    except Exception as exc:
        return {
            "scope": scope,
            "match": False,
            "sm_rms": None,
            "sm_max_dist": None,
            "ordered_pbc_rms_A": ordered_pbc_rms(numbers, pred, ref, cell, mask),
            "n_atoms": int(mask.sum()),
            "error": str(exc),
        }
    return {
        "scope": scope,
        "match": match,
        "sm_rms": sm_rms,
        "sm_max_dist": sm_max,
        "ordered_pbc_rms_A": ordered_pbc_rms(numbers, pred, ref, cell, mask),
        "n_atoms": int(mask.sum()),
    }


@torch.no_grad()
def evaluate(args) -> None:
    selection = build_or_load_selection(args)
    model_spec = MODELS[args.model_label]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, flow_cfg, ck = load_model(model_spec["ckpt"], device)
    rows = selection[args.split]
    ds = make_dataset(args, args.split, rows, model, model_spec)
    samples = seeded_samples(ds, rows, args.seed)
    force_model = None
    if bool(getattr(model.cfg, "use_langevin_param", False)):
        force_model = UMAForce(model_name=args.langevin_uma_model, task_name=args.langevin_uma_task, device=str(device))

    sm = StructureMatcher(
        ltol=args.ltol,
        stol=args.stol,
        angle_tol=args.angle_tol,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
    )
    out_rows = []
    for start in range(0, len(samples), args.batch_size):
        chunk = samples[start:start + args.batch_size]
        batch = collate_displacement(chunk)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        movable = runtime_movable_mask(model, batch)
        use_ads_ref = bool(getattr(model.cfg, "use_ads_ref_pos", False))

        def fwd(x_t, t):
            extra = {}
            if use_ads_ref:
                extra["ads_ref_pos"] = batch["ads_ref_pos"]
            if force_model is not None:
                extra["mlip_force"] = force_model(
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
            num_steps=args.flow_steps,
            use_sde=False,
            refine_final=False,
        )
        pred_cpu = pred.detach().cpu().numpy()
        ref_cpu = batch["pos_relaxed"].detach().cpu().numpy()
        nums_cpu = batch["atomic_numbers"].detach().cpu().numpy()
        tags_cpu = batch["tags"].detach().cpu().numpy()
        pad_cpu = batch["pad_mask"].detach().cpu().numpy().astype(bool)
        cell_cpu = batch["cell"].detach().cpu().numpy()
        for j, sample in enumerate(chunk):
            rec = {
                "model": args.model_label,
                "split": args.split,
                "raw_idx": int(sample["_raw_idx"]),
                "system_key": str(sample["_system_key"]),
                "sid": int(sample["sid"].item()) if torch.is_tensor(sample["sid"]) else int(sample["sid"]),
                "ads_id": int(sample["ads_id"].item()) if torch.is_tensor(sample["ads_id"]) else int(sample["ads_id"]),
                "n_atoms": int(pad_cpu[j].sum()),
                "ckpt": str(model_spec["ckpt"]),
                "ckpt_epoch": ck.get("epoch"),
                "flow_steps": int(args.flow_steps),
                "slab_source": str(model_spec["slab_source"]),
                "scores": [],
            }
            for scope in args.scopes:
                rec["scores"].append(score_structure(
                    sm,
                    nums_cpu[j],
                    pred_cpu[j],
                    ref_cpu[j],
                    cell_cpu[j],
                    tags_cpu[j],
                    pad_cpu[j],
                    scope,
                ))
            out_rows.append(rec)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_label}_{args.split}.json"
    out_path.write_text(json.dumps({
        "model": args.model_label,
        "split": args.split,
        "rows": out_rows,
        "settings": vars(args),
    }, indent=2, sort_keys=True))
    print(out_path, flush=True)


def _stats(vals: list[float]) -> dict:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None}
    arr = np.asarray(vals, dtype=float)
    return {"n": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr))}


def merge(args) -> None:
    out_dir = Path(args.out_dir)
    rows = []
    for label in MODELS:
        for split in ("train", "ood"):
            p = out_dir / f"{label}_{split}.json"
            if not p.exists():
                raise FileNotFoundError(p)
            rows.extend(json.loads(p.read_text())["rows"])
    grouped = defaultdict(list)
    for rec in rows:
        for score in rec["scores"]:
            grouped[(rec["model"], rec["split"], score["scope"])].append(score)
    summary = []
    for (model, split, scope), ss in sorted(grouped.items()):
        n = len(ss)
        summary.append({
            "model": model,
            "split": split,
            "scope": scope,
            "n": n,
            "match_rate": sum(1 for s in ss if s.get("match")) / max(n, 1),
            "sm_rms_matched": _stats([s.get("sm_rms") for s in ss if s.get("match")]),
            "ordered_pbc_rms_A_all": _stats([s.get("ordered_pbc_rms_A") for s in ss]),
            "n_errors": sum(1 for s in ss if s.get("error")),
        })
    payload = {"summary": summary, "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "merge"], default="eval")
    ap.add_argument("--model-label", choices=sorted(MODELS), default="base")
    ap.add_argument("--split", choices=["train", "ood"], default="train")
    ap.add_argument("--out-dir", default="/home1/irteam/data/replay/structure_matcher_rmsd_20260614")
    ap.add_argument("--selection", default="/home1/irteam/data/replay/structure_matcher_rmsd_20260614/selection.json")
    ap.add_argument("--rebuild-selection", action="store_true")
    ap.add_argument("--train-lmdb", default="/home/irteam/data/processed_ID/is2res_train.lmdb")
    ap.add_argument("--ood-lmdb", default="/home/irteam/data/processed_old/oc20dense.lmdb")
    ap.add_argument("--split-membership", default="/home1/irteam/data/replay/oc20dense_oc20_split_membership.json")
    ap.add_argument("--num-systems", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260614)
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--adsorbates-pkl", default="/home1/irteam/micromamba/envs/adsorbgen/lib/python3.11/site-packages/fairchem/data/oc/databases/pkls/adsorbates.pkl")
    ap.add_argument("--train-pristine-slabs", default="/home1/irteam/data-vol1/minkyu/results/pristine_slabs/is2res.pkl")
    ap.add_argument("--train-pristine-index", default="/home1/irteam/data-vol1/minkyu/results/pristine_slabs/is2res.sid_index.pkl")
    ap.add_argument("--ood-pristine-slabs", default="/home1/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense_uma.pkl")
    ap.add_argument("--ood-pristine-index", default="/home1/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense.system_index.pkl")
    ap.add_argument("--langevin-uma-model", default="uma-s-1p2")
    ap.add_argument("--langevin-uma-task", default="oc20")
    ap.add_argument("--ltol", type=float, default=0.3)
    ap.add_argument("--stol", type=float, default=0.5)
    ap.add_argument("--angle-tol", type=float, default=10.0)
    ap.add_argument("--scopes", nargs="+", default=["all", "movable", "ads"])
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.mode == "merge":
        merge(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
