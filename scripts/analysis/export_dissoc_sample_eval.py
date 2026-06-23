#!/usr/bin/env python
"""Export dissociated validation sample-eval cases as viewable structures.

This intentionally mirrors ``AdsorbGenModule._accumulate_sample_eval`` but
writes per-sample records, anomaly flags, and PDB artifacts to disk.  Output is
compatible with the existing ``viz/`` replay viewer layout:

    <out>/ep{checkpoint_epoch}/
      sys_000/
        x0.pdb          initial/random placement
        x1_flow.pdb     model prediction
        x1_relaxed.pdb  ground-truth relaxed structure
        compare.xyz     three-frame comparison
        meta.json
      _index.json
      summary.json
      dissoc_cases.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.io import write as ase_write
from ase.neighborlist import NeighborList, get_connectivity_matrix, natural_cutoffs
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement  # noqa: E402
from adsorbgen.evaluation.metrics import compute_anomaly_metrics  # noqa: E402
from adsorbgen.flow import euler_sample  # noqa: E402
from adsorbgen.replay.viz import reference_center_translation, save_structure_pdb  # noqa: E402
from adsorbgen.training.train_cli import AdsorbGenModule  # noqa: E402


def _jsonify(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, torch.Tensor):
        return _jsonify(v.detach().cpu().numpy())
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonify(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    return v


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def _atoms(numbers, positions, cell, tags) -> Atoms:
    atoms = Atoms(
        numbers=np.asarray(numbers, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.float64),
        cell=np.asarray(cell, dtype=np.float64),
        pbc=True,
    )
    atoms.set_tags(np.asarray(tags, dtype=np.int64).tolist())
    return atoms


def _connectivity(atoms: Atoms) -> np.ndarray:
    cutoffs = natural_cutoffs(atoms, mult=1.0)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    return get_connectivity_matrix(nl.nl).toarray().astype(np.int8)


def _canonical_ads_atoms(record: dict[str, Any], ads_db: dict[int, Any]) -> Atoms | None:
    ads_id = int(record.get("ads_id", -1))
    if ads_id < 0:
        return None
    entry = ads_db.get(ads_id)
    if entry is None:
        return None
    canon = entry[0]
    tags = record["tags"].numpy()
    ads_mask = tags == 2
    rec_z = record["atomic_numbers"].numpy()[ads_mask].astype(np.int64)
    canon_z = np.asarray(canon.get_atomic_numbers(), dtype=np.int64)
    if canon_z.shape != rec_z.shape or not np.array_equal(canon_z, rec_z):
        return None
    return Atoms(
        numbers=canon_z,
        positions=canon.get_positions(),
        cell=record["cell"].numpy(),
        pbc=True,
    )


def _bond_change_summary(record: dict[str, Any], ads_db: dict[int, Any]) -> dict[str, Any]:
    tags = record["tags"].numpy()
    ads_mask = tags == 2
    ads_z = record["atomic_numbers"].numpy()[ads_mask].astype(np.int64)
    pred_pos = record["pos_pred"].numpy()[ads_mask]
    pred_atoms = Atoms(
        numbers=ads_z,
        positions=pred_pos,
        cell=record["cell"].numpy(),
        pbc=True,
    )
    ref_atoms = _canonical_ads_atoms(record, ads_db)
    ref_source = "canonical_adsorbates_pkl"
    if ref_atoms is None:
        ref_source = "pos_ref_fallback"
        ref_atoms = Atoms(
            numbers=ads_z,
            positions=record["pos_ref"].numpy()[ads_mask],
            cell=record["cell"].numpy(),
            pbc=True,
        )

    ref_conn = _connectivity(ref_atoms)
    pred_conn = _connectivity(pred_atoms)
    broken = []
    formed = []
    for i in range(len(ads_z)):
        for j in range(i + 1, len(ads_z)):
            if ref_conn[i, j] and not pred_conn[i, j]:
                broken.append({
                    "i": i,
                    "j": j,
                    "Zi": int(ads_z[i]),
                    "Zj": int(ads_z[j]),
                    "pred_dist_A": float(pred_atoms.get_distance(i, j, mic=True)),
                    "ref_dist_A": float(ref_atoms.get_distance(i, j, mic=True)),
                })
            if pred_conn[i, j] and not ref_conn[i, j]:
                formed.append({
                    "i": i,
                    "j": j,
                    "Zi": int(ads_z[i]),
                    "Zj": int(ads_z[j]),
                    "pred_dist_A": float(pred_atoms.get_distance(i, j, mic=True)),
                    "ref_dist_A": float(ref_atoms.get_distance(i, j, mic=True)),
                })
    return {
        "connectivity_reference": ref_source,
        "n_ref_bonds": int(np.triu(ref_conn, 1).sum()),
        "n_pred_bonds": int(np.triu(pred_conn, 1).sum()),
        "n_broken_bonds": len(broken),
        "n_formed_bonds": len(formed),
        "broken_bonds": broken,
        "formed_bonds": formed,
    }


def _save_compare_xyz(record: dict[str, Any], out_path: Path, offset: np.ndarray) -> None:
    numbers = record["atomic_numbers"].numpy()
    tags = record["tags"].numpy()
    cell = record["cell"].numpy()
    frames = [
        _atoms(numbers, record["pos_ref"].numpy() + offset, cell, tags),
        _atoms(numbers, record["pos_pred"].numpy() + offset, cell, tags),
        _atoms(numbers, record["pos_gt"].numpy() + offset, cell, tags),
    ]
    frames[0].info["label"] = "x0"
    frames[1].info["label"] = "x1_flow"
    frames[2].info["label"] = "x1_relaxed_gt"
    ase_write(str(out_path), frames, format="extxyz")


@torch.no_grad()
def generate_records(args, model: AdsorbGenModule, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    model.to(device)
    cfg = model._model_cfg()
    use_self_cond = bool(getattr(cfg, "use_self_cond", False))
    use_ads_ref = bool(getattr(cfg, "use_ads_ref_pos", False))
    langevin_force_model = model._get_langevin_force_model()
    records: list[dict[str, Any]] = []
    t0 = time.time()

    for batch_idx, batch_cpu in enumerate(loader):
        if args.max_samples > 0 and len(records) >= args.max_samples:
            break
        batch = _to_device(batch_cpu, device)
        movable = model._effective_movable_mask(batch)
        state = {"prev_pred": None}

        def model_forward(x_t, t):
            extra = {}
            if use_self_cond:
                extra["prev_pred"] = state["prev_pred"]
            if use_ads_ref:
                extra["ads_ref_pos"] = batch["ads_ref_pos"]
            if langevin_force_model is not None:
                extra["mlip_force"] = langevin_force_model(
                    x_t.detach(),
                    batch["cell"],
                    batch["atomic_numbers"],
                    batch["pad_mask"],
                )
                extra["langevin_prediction_type"] = model.flow_cfg.prediction_type
            out = model.model(
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
            if use_self_cond:
                state["prev_pred"] = out.detach()
            return out

        autocast_enabled = device.type == "cuda" and args.precision in {"bf16", "bf16-mixed"}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            x_out = euler_sample(
                model_forward,
                batch["pos"],
                movable,
                batch["pad_mask"],
                model.flow_cfg,
                num_steps=args.sample_steps,
            )
        B = int(batch["pos"].shape[0])
        for i in range(B):
            if args.max_samples > 0 and len(records) >= args.max_samples:
                break
            n = int(batch["pad_mask"][i].sum().item())
            cell_i = batch["cell"][i].detach().cpu()
            if cell_i.dim() == 3:
                cell_i = cell_i[0]
            records.append({
                "sample_idx": len(records),
                "sid": int(batch["sid"][i].item()) if "sid" in batch else -1,
                "system_key": batch.get("system_key", [None] * B)[i],
                "config_key": batch.get("config_key", [None] * B)[i],
                "ads_id": int(batch["ads_id"][i].item()) if "ads_id" in batch else -1,
                "pos_pred": x_out[i, :n].detach().cpu(),
                "pos_gt": batch["pos_relaxed"][i, :n].detach().cpu(),
                "pos_ref": batch["pos"][i, :n].detach().cpu(),
                "movable_mask": movable[i, :n].detach().cpu(),
                "atomic_numbers": batch["atomic_numbers"][i, :n].detach().cpu(),
                "tags": batch["tags"][i, :n].detach().cpu(),
                "cell": cell_i,
            })
        if args.progress_every > 0 and (batch_idx + 1) % args.progress_every == 0:
            elapsed = time.time() - t0
            print(
                f"[sample] batches={batch_idx + 1} records={len(records)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-lmdb", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "bf16-mixed"])
    p.add_argument("--max-viz", type=int, default=128)
    p.add_argument("--prior-mode", default="random_heuristic")
    p.add_argument("--slab-source", default="pristine_relaxed")
    p.add_argument("--val-pristine-slabs", default="/home1/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense_uma.pkl")
    p.add_argument("--val-pristine-index", default="/home1/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense.system_index.pkl")
    p.add_argument("--adsorbates-pkl", default=os.environ.get("ADSORBATES_PKL", "/home/irteam/data/pkls/adsorbates.pkl"))
    p.add_argument("--anomaly-workers", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=10)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    epoch = int(blob.get("epoch", -1))
    hparams = blob.get("hyper_parameters", {})
    model_cfg = hparams.get("model_cfg")
    provide_ref = bool(getattr(model_cfg, "use_ads_ref_pos", False))
    print(f"[load] ckpt={ckpt_path} epoch={epoch} provide_ads_ref_pos={provide_ref}", flush=True)

    ds = PlacementPriorDataset(
        args.val_lmdb,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        training_aug=False,
        unique_by_system_key=True,
        prior_mode=args.prior_mode,
        provide_ads_ref_pos=provide_ref,
        adsorbates_pkl=args.adsorbates_pkl,
        slab_source=args.slab_source,
        pristine_slabs=args.val_pristine_slabs,
        pristine_index=args.val_pristine_index,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_displacement,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = AdsorbGenModule.load_from_checkpoint(
        str(ckpt_path),
        map_location="cpu",
        weights_only=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = generate_records(args, model, loader, device)
    print(f"[score] generated_records={len(records)}", flush=True)

    strict = compute_anomaly_metrics(
        records,
        workers=args.anomaly_workers,
        pristine_slabs=Path(args.val_pristine_slabs) if args.val_pristine_slabs else None,
        pristine_sid_index=Path(args.val_pristine_index) if args.val_pristine_index else None,
    )
    per = strict["per_sample"]
    agg = strict["aggregate"]
    by_idx = {int(x["sid"]) if x.get("sid") is not None else i: x for i, x in enumerate(per)}

    # ``sid`` is not guaranteed unique in all datasets, so attach by list order.
    dissoc_indices = [i for i, x in enumerate(per) if x.get("has_dissoc") is True]
    all_ads_counts = np.asarray([int((r["tags"] == 2).sum().item()) for r in records], dtype=np.int64)
    dissoc_ads_counts = all_ads_counts[dissoc_indices] if dissoc_indices else np.asarray([], dtype=np.int64)
    non_dissoc_ads_counts = np.delete(all_ads_counts, dissoc_indices) if dissoc_indices else all_ads_counts

    with open(args.adsorbates_pkl, "rb") as f:
        ads_db = pickle.load(f)

    ep_dir = out_root / f"ep{epoch if epoch >= 0 else 'unknown'}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    systems = []
    csv_rows = []
    for out_i, rec_i in enumerate(dissoc_indices[: max(args.max_viz, 0)]):
        record = records[rec_i]
        flags = per[rec_i]
        sys_dir_name = f"sys_{out_i:03d}"
        sys_dir = ep_dir / sys_dir_name
        sys_dir.mkdir(parents=True, exist_ok=True)

        numbers = record["atomic_numbers"].numpy()
        tags = record["tags"].numpy()
        cell = record["cell"].numpy()
        offset = reference_center_translation(record["pos_ref"].numpy(), cell)
        save_structure_pdb(numbers, record["pos_ref"].numpy(), cell, tags, sys_dir / "x0.pdb", offset=offset)
        save_structure_pdb(numbers, record["pos_pred"].numpy(), cell, tags, sys_dir / "x1_flow.pdb", offset=offset)
        save_structure_pdb(numbers, record["pos_gt"].numpy(), cell, tags, sys_dir / "x1_relaxed.pdb", offset=offset)
        _save_compare_xyz(record, sys_dir / "compare.xyz", offset=offset)

        bond_summary = _bond_change_summary(record, ads_db)
        n_ads = int((record["tags"] == 2).sum().item())
        meta = {
            "global_idx": out_i,
            "source_sample_idx": rec_i,
            "sys_dir_name": sys_dir_name,
            "status": "dissociated",
            "sid": int(record["sid"]),
            "system_key": record.get("system_key"),
            "config_key": record.get("config_key"),
            "ads_id": int(record["ads_id"]),
            "n_atoms": int(record["tags"].numel()),
            "n_ads_atoms": n_ads,
            "has_dissoc": True,
            "anomaly_flags": flags,
            **bond_summary,
        }
        (sys_dir / "meta.json").write_text(json.dumps(_jsonify(meta), indent=2))
        systems.append({
            "global_idx": out_i,
            "source_sample_idx": rec_i,
            "sys_dir_name": sys_dir_name,
            "sid": int(record["sid"]),
            "ads_id": int(record["ads_id"]),
            "n_atoms": int(record["tags"].numel()),
            "n_ads_atoms": n_ads,
            "status": "dissociated",
            "has_dissoc": True,
            "n_broken_bonds": int(bond_summary["n_broken_bonds"]),
            "n_formed_bonds": int(bond_summary["n_formed_bonds"]),
        })
        csv_rows.append(systems[-1])

    index_payload = {
        "epoch": epoch,
        "epoch_dir": ep_dir.name,
        "n_systems": len(systems),
        "n_dissoc_total": len(dissoc_indices),
        "systems": systems,
        "meta": {
            "ckpt": str(ckpt_path),
            "val_lmdb": args.val_lmdb,
            "sample_steps": args.sample_steps,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
        },
    }
    (ep_dir / "_index.json").write_text(json.dumps(_jsonify(index_payload), indent=2))

    summary = {
        "aggregate": agg,
        "n_records": len(records),
        "n_dissoc": len(dissoc_indices),
        "dissoc_rate": len(dissoc_indices) / max(len(records), 1),
        "n_viz_exported": len(systems),
        "ads_atom_count": {
            "all_mean": float(all_ads_counts.mean()) if all_ads_counts.size else None,
            "all_median": float(np.median(all_ads_counts)) if all_ads_counts.size else None,
            "dissoc_mean": float(dissoc_ads_counts.mean()) if dissoc_ads_counts.size else None,
            "dissoc_median": float(np.median(dissoc_ads_counts)) if dissoc_ads_counts.size else None,
            "non_dissoc_mean": float(non_dissoc_ads_counts.mean()) if non_dissoc_ads_counts.size else None,
            "non_dissoc_median": float(np.median(non_dissoc_ads_counts)) if non_dissoc_ads_counts.size else None,
        },
        "paths": {
            "viz_root": str(out_root),
            "epoch_dir": str(ep_dir),
            "index": str(ep_dir / "_index.json"),
        },
    }
    (ep_dir / "summary.json").write_text(json.dumps(_jsonify(summary), indent=2))
    with open(ep_dir / "dissoc_cases.csv", "w", newline="") as f:
        fieldnames = [
            "global_idx",
            "source_sample_idx",
            "sys_dir_name",
            "sid",
            "ads_id",
            "n_atoms",
            "n_ads_atoms",
            "status",
            "has_dissoc",
            "n_broken_bonds",
            "n_formed_bonds",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("[done]", json.dumps(_jsonify(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
