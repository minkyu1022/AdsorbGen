#!/usr/bin/env python
"""Evaluate UMA single-point energy-gap curves across checkpoints.

The evaluator mirrors training sample_eval on the Dense validation split:
fresh random_heuristic placement, deterministic seeds, Euler flow sampling,
then UMA single-point E_sys(pred) - reference E_sys.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import lmdb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO = Path(os.environ.get("ADSGEN_ROOT", "/home1/irteam/AdsorbGen")).resolve()
GEOOPT = REPO / "geoopt"
for p in (REPO, GEOOPT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement  # noqa: E402
from adsorbgen.evaluation.energy import UMAEnergy, UMAForce  # noqa: E402
from adsorbgen.flow import euler_sample  # noqa: E402
from adsorbgen.replay.eval import _model_cfg, _runtime_movable_mask  # noqa: E402
from geoopt import load_model_from_ckpt  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_dir: Path
    slab_source: str
    pristine_slabs: str = ""
    pristine_index: str = ""


RUNS = (
    RunSpec(
        label="ID_pairdist",
        run_dir=Path("/home1/irteam/runs/training/ID_mlip_pairdist_only_1x_bs64_expand_20260521_213544"),
        slab_source="initial",
    ),
    RunSpec(
        label="x1_LP",
        run_dir=Path("/home1/irteam/runs/training/x1_LP_102M"),
        slab_source="initial",
    ),
    RunSpec(
        label="x1_LP_bare_slab",
        run_dir=Path("/home1/irteam/runs/training/x1_LP_102M_relaxed_bare_slab_x0"),
        slab_source="pristine_relaxed",
        pristine_slabs="/home1/irteam/data-vol1/minkyu/results/pristine_slabs/is2res.pkl",
        pristine_index="/home1/irteam/data-vol1/minkyu/results/pristine_slabs/is2res.sid_index.pkl",
    ),
)


class TrainUniquePlacementDataset(torch.utils.data.Dataset):
    """Fixed train-system subset with repeated fresh placements per system."""

    def __init__(self, rows: list[dict[str, Any]], args, use_ads_ref: bool):
        self.rows = rows
        self.num_placements = int(args.num_placements)
        self.seed = int(args.selection_seed)
        self.datasets = {}
        self.envs = {}
        for path in sorted({str(r["lmdb"]) for r in rows}):
            self.datasets[path] = PlacementPriorDataset(
                path,
                prior_mode=args.prior_mode,
                max_samples=None,
                training_aug=False,
                provide_ads_ref_pos=use_ads_ref,
                skip_anomaly=False,
                slab_source=str(args.eval_slab_source),
                pristine_slabs=str(args.eval_pristine_slabs or ""),
                pristine_index=str(args.eval_pristine_index or ""),
            )
            self.envs[path] = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False)

    def __len__(self) -> int:
        return len(self.rows) * self.num_placements

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base_i = int(idx) // self.num_placements
        placement_i = int(idx) % self.num_placements
        row = self.rows[base_i]
        path = str(row["lmdb"])
        raw_idx = int(row["raw_idx"])
        draw_seed = (self.seed + base_i * self.num_placements + placement_i) & 0xFFFF_FFFF
        random.seed(draw_seed)
        np.random.seed(draw_seed)
        torch.manual_seed(draw_seed)
        sample = self.datasets[path][raw_idx]
        # For this train-fitting curve, y_relaxed is repurposed as E_sys_ref.
        sample["y_relaxed"] = torch.tensor(float(row["E_sys_ref"]), dtype=torch.float32)
        return sample


def _read_lmdb_entry(env: lmdb.Environment, idx: int) -> dict[str, Any]:
    with env.begin() as txn:
        raw = txn.get(str(int(idx)).encode("ascii"))
    if raw is None:
        raise KeyError(f"missing LMDB key {idx}")
    return pickle.loads(raw)


def _build_train_unique_selection(args, out_dir: Path) -> list[dict[str, Any]]:
    out_path = out_dir / "selected_train_unique_1000.json"
    if out_path.exists():
        return json.loads(out_path.read_text())["rows"]

    rng = np.random.default_rng(int(args.selection_seed))
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    lmdb_paths = [str(p) for p in args.train_lmdb]
    for lmdb_id, lmdb_path in enumerate(lmdb_paths):
        env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            n_total = int(pickle.loads(txn.get(b"length")))
            mask_raw = txn.get(b"anomaly_mask")
        clean_mask = np.ones(n_total, dtype=bool)
        if mask_raw is not None:
            clean_mask = np.asarray(pickle.loads(mask_raw), dtype=np.int8)[:n_total] == 0
        with env.begin() as txn:
            for raw_idx in range(n_total):
                if not bool(clean_mask[raw_idx]):
                    continue
                raw = txn.get(str(raw_idx).encode("ascii"))
                if raw is None:
                    continue
                e = pickle.loads(raw)
                sid = int(e.get("sid", -1))
                key = str(e.get("system_key", f"sid:{sid}" if sid >= 0 else f"{lmdb_id}:{raw_idx}"))
                if key in seen:
                    continue
                ref = e.get("mlip_e_total")
                if ref is None or not math.isfinite(float(ref)):
                    continue
                seen.add(key)
                candidates.append({
                    "lmdb_id": int(lmdb_id),
                    "lmdb": lmdb_path,
                    "raw_idx": int(raw_idx),
                    "sid": int(sid),
                    "system_key": key,
                    "ads_id": int(e.get("ads_id", -1)),
                    "n_atoms": int(np.asarray(e["atomic_numbers"]).shape[0]),
                    "E_sys_ref": float(ref),
                })
        env.close()
    if len(candidates) < int(args.max_samples):
        raise RuntimeError(f"only {len(candidates)} train unique candidates, need {args.max_samples}")
    idx = rng.choice(len(candidates), size=int(args.max_samples), replace=False)
    rows = [candidates[int(i)] for i in idx]
    rows.sort(key=lambda r: (r["lmdb_id"], r["raw_idx"]))
    payload = {
        "dataset": "train_unique",
        "selection_seed": int(args.selection_seed),
        "max_samples": int(args.max_samples),
        "num_candidates": int(len(candidates)),
        "num_selected": int(len(rows)),
        "train_lmdb": lmdb_paths,
        "reference": "mlip_e_total",
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({k: payload[k] for k in payload if k != "rows"}, indent=2), flush=True)
    return rows


def _load_refs(path: Path) -> dict[str, float]:
    if path.is_dir():
        if (path / "gt_results" / "global_minima.json").exists():
            path = path / "gt_results" / "global_minima.json"
        elif (path / "oc20dense_mlip_global_min_by_system.pkl").exists():
            path = path / "oc20dense_mlip_global_min_by_system.pkl"
        else:
            raise FileNotFoundError(f"no supported reference file under {path}")
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            raw = pickle.load(f)
    else:
        with path.open() as f:
            raw = json.load(f)
    refs: dict[str, float] = {}
    items = raw.values() if isinstance(raw, dict) else raw
    for rec in items:
        if not isinstance(rec, dict):
            continue
        system_key = rec.get("system_id") or rec.get("system_key")
        energy = rec.get("e_sys_relaxed", rec.get("E_sys_min"))
        if system_key is not None and energy is not None:
            refs[str(system_key)] = float(energy)
    if not refs:
        raise RuntimeError(f"no refs loaded from {path}")
    return refs


def _epoch_from_ckpt(path: Path) -> int:
    m = re.search(r"epoch=(\d+)", path.name)
    if not m:
        raise ValueError(f"cannot parse epoch from {path.name}")
    return int(m.group(1))


def _ckpts(run: RunSpec, max_epoch: int) -> list[Path]:
    out = []
    for p in run.run_dir.glob("ckpt_epochepoch=*.ckpt"):
        ep = _epoch_from_ckpt(p)
        if ep <= max_epoch:
            out.append(p)
    return sorted(out, key=_epoch_from_ckpt)


def _tasks(max_epoch: int, seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for run in RUNS:
        for ckpt in _ckpts(run, max_epoch):
            ep = _epoch_from_ckpt(ckpt)
            for seed in seeds:
                rows.append({
                    "label": run.label,
                    "run_dir": str(run.run_dir),
                    "ckpt": str(ckpt),
                    "epoch": ep,
                    "seed": int(seed),
                    "slab_source": run.slab_source,
                    "pristine_slabs": run.pristine_slabs,
                    "pristine_index": run.pristine_index,
                })
    return rows


def _stats(vals: list[float]) -> dict[str, float | int | None]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "mae": None, "rmse": None, "median": None, "p90": None, "p95": None}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "mae": float(np.abs(arr).mean()),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def _refs_for_batch(batch: dict[str, Any], refs: dict[str, float] | None) -> torch.Tensor:
    if refs is None:
        return batch["y_relaxed"].detach().cpu().float()
    keys = batch.get("system_key")
    if keys is None:
        raise KeyError("batch has no system_key")
    vals = []
    missing = []
    for k in keys:
        sk = str(k)
        ref = refs.get(sk)
        if ref is None:
            missing.append(sk)
            vals.append(float("nan"))
        else:
            vals.append(float(ref))
    if missing:
        print(f"[refs] missing {len(missing)} refs; first={missing[:3]}", flush=True)
    return torch.tensor(vals, dtype=torch.float32)


@torch.no_grad()
def evaluate_one(task: dict[str, Any], args) -> dict[str, Any]:
    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for UMA energy evaluation")

    refs = None if args.dataset == "train_unique" else _load_refs(Path(args.ref_path))
    model, flow_cfg = load_model_from_ckpt(Path(task["ckpt"]), device)
    cfg = _model_cfg(model)
    use_ads_ref = bool(getattr(cfg, "use_ads_ref_pos", False))
    langevin_force_model = None
    if bool(getattr(cfg, "use_langevin_param", False)):
        if str(getattr(cfg, "langevin_eval_on", "x_t")) != "x_t":
            raise ValueError("Only langevin_eval_on='x_t' is implemented")
        langevin_force_model = UMAForce(
            model_name=args.langevin_uma_model,
            task_name=args.langevin_uma_task,
            device=str(device),
        )

    if args.dataset == "train_unique":
        args.eval_slab_source = str(task["slab_source"])
        args.eval_pristine_slabs = str(task.get("pristine_slabs") or "")
        args.eval_pristine_index = str(task.get("pristine_index") or "")
        rows = _build_train_unique_selection(args, Path(args.out_dir))
        ds = TrainUniquePlacementDataset(rows, args, use_ads_ref=use_ads_ref)
    else:
        ds = PlacementPriorDataset(
            args.lmdb,
            max_samples=int(args.max_samples),
            training_aug=False,
            unique_by_system_key=True,
            prior_mode=args.prior_mode,
            provide_ads_ref_pos=use_ads_ref,
            skip_anomaly=False,
            slab_source=str(task["slab_source"]),
            pristine_slabs=str(task.get("pristine_slabs") or ""),
            pristine_index=str(task.get("pristine_index") or ""),
        )
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_displacement,
        pin_memory=True,
        drop_last=False,
    )
    energy = UMAEnergy(
        model_name=args.uma_model,
        task_name=args.uma_task,
        device=str(device),
        normalize_per_atom=False,
    )

    deltas: list[float] = []
    n_missing = 0
    for batch in tqdm(
        loader,
        desc=f"{task['label']} ep{task['epoch']:03d} seed{seed}",
        dynamic_ncols=True,
        leave=False,
    ):
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

        x_out = euler_sample(
            fwd,
            batch["pos"],
            movable,
            batch["pad_mask"],
            flow_cfg,
            num_steps=int(args.flow_steps),
            use_sde=False,
            refine_final=False,
        )
        e_pred = energy(x_out, batch["cell"], batch["atomic_numbers"], batch["pad_mask"])
        ref = _refs_for_batch(batch, refs).to(device)
        delta = (e_pred.detach().float() - ref).cpu().numpy()
        n_missing += int(np.isnan(delta).sum())
        deltas.extend(float(x) for x in delta if np.isfinite(x))

    st = _stats(deltas)
    row = {
        **task,
        "n_missing_ref": int(n_missing),
        "n_eval": int(st["n"] or 0),
        "dataset": str(args.dataset),
        "num_placements": int(args.num_placements),
        "delta_E_sys_mean_eV": st["mean"],
        "delta_E_sys_mae_eV": st["mae"],
        "delta_E_sys_rmse_eV": st["rmse"],
        "delta_E_sys_median_eV": st["median"],
        "delta_E_sys_p90_eV": st["p90"],
        "delta_E_sys_p95_eV": st["p95"],
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def merge_and_plot(out_dir: Path) -> None:
    rows = []
    for p in sorted((out_dir / "per_task").glob("*.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        raise RuntimeError(f"no per-task results under {out_dir / 'per_task'}")
    _write_csv(out_dir / "energy_gap_per_seed.csv", rows)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((str(r["label"]), int(r["epoch"])), []).append(r)
    agg = []
    for (label, epoch), vals in sorted(grouped.items()):
        means = np.asarray([float(v["delta_E_sys_mean_eV"]) for v in vals], dtype=np.float64)
        maes = np.asarray([float(v["delta_E_sys_mae_eV"]) for v in vals], dtype=np.float64)
        agg.append({
            "label": label,
            "epoch": epoch,
            "n_seeds": int(len(vals)),
            "mean_eV_seed_avg": float(means.mean()),
            "mean_eV_seed_std": float(means.std(ddof=0)),
            "mae_eV_seed_avg": float(maes.mean()),
            "mae_eV_seed_std": float(maes.std(ddof=0)),
            "n_eval_total": int(sum(int(v["n_eval"]) for v in vals)),
        })
    _write_csv(out_dir / "energy_gap_curve.csv", agg)
    (out_dir / "energy_gap_curve.json").write_text(json.dumps(agg, indent=2, sort_keys=True))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "ID_pairdist": "#1f77b4",
        "x1_LP": "#d62728",
        "x1_LP_bare_slab": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for label in [r.label for r in RUNS]:
        vals = [r for r in agg if r["label"] == label]
        if not vals:
            continue
        x = np.asarray([v["epoch"] + 1 for v in vals], dtype=float)
        y = np.asarray([v["mean_eV_seed_avg"] for v in vals], dtype=float)
        e = np.asarray([v["mean_eV_seed_std"] for v in vals], dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, label=label, color=colors.get(label))
        ax.fill_between(x, y - e, y + e, color=colors.get(label), alpha=0.16, linewidth=0)
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("UMA-SP E_sys(pred) - reference E_sys (eV)")
    ax.set_title("Train unique energy gap curve")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_gap_curve_mean.png", dpi=180)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for label in [r.label for r in RUNS]:
        vals = [r for r in agg if r["label"] == label]
        if not vals:
            continue
        x = np.asarray([v["epoch"] + 1 for v in vals], dtype=float)
        y = np.asarray([v["mae_eV_seed_avg"] for v in vals], dtype=float)
        e = np.asarray([v["mae_eV_seed_std"] for v in vals], dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, label=label, color=colors.get(label))
        ax.fill_between(x, np.maximum(y - e, 0.0), y + e, color=colors.get(label), alpha=0.16, linewidth=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("UMA-SP |E_sys(pred) - reference E_sys| (eV)")
    ax.set_title("Train unique energy gap MAE curve")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_gap_curve_mae.png", dpi=180)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["list", "worker", "merge"], required=True)
    ap.add_argument("--dataset", choices=["dense", "train_unique"], default="train_unique")
    ap.add_argument("--worker-idx", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--max-epoch", type=int, default=99)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--lmdb", default="/home1/irteam/data/processed_old/oc20dense.lmdb")
    ap.add_argument("--train-lmdb", nargs="+", default=[
        "/home1/irteam/data/processed_ID/is2res_train.lmdb",
        "/home1/irteam/data/processed_ID/is2res_val.lmdb",
    ])
    ap.add_argument("--ref-path", default="/home1/irteam/data/replay/oc20dense_mlip_global_min_by_system.pkl")
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--num-placements", type=int, default=3)
    ap.add_argument("--selection-seed", type=int, default=20260614)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--flow-steps", type=int, default=20)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--langevin-uma-model", default="uma-s-1p2")
    ap.add_argument("--langevin-uma-task", default="oc20")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    tasks = _tasks(int(args.max_epoch), seeds)
    if args.mode == "list":
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tasks.json").write_text(json.dumps(tasks, indent=2, sort_keys=True))
        payload = {"tasks": len(tasks), "out_dir": str(out_dir), "dataset": str(args.dataset)}
        if args.dataset == "train_unique":
            rows = _build_train_unique_selection(args, out_dir)
            payload["selected_train_unique"] = len(rows)
            payload["num_placements"] = int(args.num_placements)
        print(json.dumps(payload, indent=2), flush=True)
        return
    if args.mode == "merge":
        merge_and_plot(out_dir)
        print(json.dumps({"merged": True, "out_dir": str(out_dir)}, indent=2), flush=True)
        return

    per_task = out_dir / "per_task"
    per_task.mkdir(parents=True, exist_ok=True)
    my_tasks = [t for i, t in enumerate(tasks) if i % int(args.num_workers) == int(args.worker_idx)]
    print(
        json.dumps(
            {
                "worker_idx": int(args.worker_idx),
                "num_workers": int(args.num_workers),
                "tasks": len(my_tasks),
                "out_dir": str(out_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    for task in my_tasks:
        name = f"{task['label']}_ep{int(task['epoch']):03d}_seed{int(task['seed'])}.json"
        out = per_task / name
        if out.exists():
            print(f"[skip] {out}", flush=True)
            continue
        row = evaluate_one(task, args)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(row, indent=2, sort_keys=True))
        os.replace(tmp, out)
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
