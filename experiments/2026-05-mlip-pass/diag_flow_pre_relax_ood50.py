#!/usr/bin/env python
"""Flow-only OOD50 diagnostics before UMA/LBFGS relaxation."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm.auto import tqdm

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement  # noqa: E402
from adsorbgen.evaluation.metrics import compute_anomaly_metrics  # noqa: E402
from adsorbgen.flow import FlowConfig, euler_sample  # noqa: E402
from adsorbgen.models.dit import DiTDenoiserConfig  # noqa: E402
from adsorbgen.models.dit_v2 import DiTDenoiserV2Config  # noqa: E402
from adsorbgen.models.factory import build_model  # noqa: E402
from adsorbgen.replay.eval import _model_cfg, _runtime_movable_mask  # noqa: E402


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    torch.serialization.add_safe_globals([DiTDenoiserConfig, DiTDenoiserV2Config, FlowConfig])
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    model = build_model(hp["model_cfg"])
    sd = ck["state_dict"]
    stripped = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(stripped, strict=False)
    model.adsorbgen_movable_mode = str(hp.get("movable_mode", "surface_ads"))
    model.to(device).eval()
    return model, hp["flow_cfg"], ck.get("epoch", None)


def load_selected_systems(args) -> list[str]:
    split = json.load(open(args.split_membership))
    rows = split["subset100_rows"]
    ood = sorted(r["system_key"] for r in rows if str(r["class"]).startswith("val_ood"))
    rng = np.random.default_rng(args.seed)
    return sorted(rng.choice(ood, size=args.num_systems, replace=False).tolist())


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


def build_tasks(args) -> list[tuple[int, str, int, int]]:
    selected_systems = load_selected_systems(args)
    raw_idx_by_system = map_system_to_raw_idx(args.lmdb, set(selected_systems))
    tasks = []
    for sys_i, sk in enumerate(selected_systems):
        for sample_i in range(args.num_samples):
            global_i = sys_i * args.num_samples + sample_i
            tasks.append((global_i, sk, sample_i, raw_idx_by_system[sk]))
    return tasks


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lmdb", default="/home/irteam/data/processed_old/oc20dense.lmdb")
    ap.add_argument("--split-membership", default="/home/irteam/data/replay/oc20dense_oc20_split_membership.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260520)
    ap.add_argument("--num-systems", type=int, default=50)
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--flow-batch-size", type=int, default=64)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--slab-source", default="pristine_relaxed", choices=["initial", "pristine_relaxed"])
    ap.add_argument("--placement-pristine-slabs", default="/home/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense_uma.pkl")
    ap.add_argument("--placement-pristine-index", default="/home/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense.system_index.pkl")
    ap.add_argument("--pristine-slabs", default="/home/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense_uma.pkl")
    ap.add_argument("--pristine-index", default="/home/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense.system_index.pkl")
    ap.add_argument("--anomaly-workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")

    t0 = time.time()
    model, flow_cfg, epoch = load_model_from_ckpt(Path(args.ckpt), device)
    model_cfg = _model_cfg(model)
    use_ads_ref = bool(getattr(model_cfg, "use_ads_ref_pos", False))
    langevin_force_model = None
    if bool(getattr(model_cfg, "use_langevin_param", False)):
        if str(getattr(model_cfg, "langevin_eval_on", "x_t")) != "x_t":
            raise ValueError("Only langevin_eval_on='x_t' is implemented")
        from adsorbgen.evaluation.energy import UMAForce  # noqa: WPS433

        langevin_force_model = UMAForce(device=str(device))
    ds = PlacementPriorDataset(
        args.lmdb,
        prior_mode=args.prior_mode,
        max_samples=None,
        provide_ads_ref_pos=use_ads_ref,
        skip_anomaly=False,
        slab_source=args.slab_source,
        pristine_slabs=args.placement_pristine_slabs,
        pristine_index=args.placement_pristine_index,
    )
    tasks = build_tasks(args)
    records = []
    pre_stats = []

    for start in tqdm(range(0, len(tasks), args.flow_batch_size), desc="flow-only", unit="batch"):
        chunk = tasks[start:start + args.flow_batch_size]
        samples = []
        metas = []
        for global_i, sk, sample_i, raw_idx in chunk:
            np.random.seed((args.seed + global_i) & 0xFFFF_FFFF)
            random.seed((args.seed + global_i) & 0xFFFF_FFFF)
            samples.append(ds[int(raw_idx)])
            metas.append((global_i, sk, sample_i, raw_idx))

        batch = collate_displacement(samples)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        movable = _runtime_movable_mask(model, batch)

        def fwd(x_t, t, _b=batch):
            extra = {}
            if use_ads_ref:
                extra["ads_ref_pos"] = _b["ads_ref_pos"]
            if langevin_force_model is not None:
                extra["mlip_force"] = langevin_force_model(
                    x_t.detach(),
                    _b["cell"],
                    _b["atomic_numbers"],
                    _b["pad_mask"],
                )
                extra["langevin_prediction_type"] = flow_cfg.prediction_type
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

        x_out = euler_sample(fwd, batch["pos"], movable, batch["pad_mask"], flow_cfg, num_steps=args.flow_steps)
        for i, (global_i, sk, sample_i, raw_idx) in enumerate(metas):
            n = int(batch["pad_mask"][i].sum().item())
            tags = batch["tags"][i, :n].detach().cpu()
            cell = batch["cell"][i].detach().cpu()
            if cell.dim() == 3:
                cell = cell[0]
            pos_ref = batch["pos"][i, :n].detach().cpu()
            pos_gt = batch["pos_relaxed"][i, :n].detach().cpu()
            pos_pred = x_out[i, :n].detach().cpu()
            mov = movable[i, :n].detach().cpu().bool()
            ads_mask = tags == 2
            surf_mask = mov & (tags == 1)
            ads_mov = mov & ads_mask

            def mae(mask):
                if not mask.any():
                    return None
                return float((pos_pred[mask] - pos_gt[mask]).norm(dim=-1).mean().item())

            rec = {
                "global_i": int(global_i),
                "system_key": sk,
                "sample_i": int(sample_i),
                "raw_idx": int(raw_idx),
                "ads_id": int(batch["ads_id"][i].item()),
                "sid": -1,
                "pos_ref": pos_ref,
                "pos_pred": pos_pred,
                "pos_gt": pos_gt,
                "movable_mask": mov,
                "atomic_numbers": batch["atomic_numbers"][i, :n].detach().cpu(),
                "tags": tags,
                "cell": cell,
            }
            records.append(rec)
            pre_stats.append({
                "global_i": int(global_i),
                "system_key": sk,
                "sample_i": int(sample_i),
                "raw_idx": int(raw_idx),
                "ads_id": int(batch["ads_id"][i].item()),
                "n_atoms": int(n),
                "n_ads_atoms": int(ads_mask.sum().item()),
                "mae_movable_A": mae(mov),
                "mae_ads_A": mae(ads_mov),
                "mae_surface_A": mae(surf_mask),
            })

    strict = compute_anomaly_metrics(
        records,
        workers=args.anomaly_workers,
        pristine_slabs=Path(args.pristine_slabs),
        pristine_sid_index=Path(args.pristine_index),
    )
    per = strict["per_sample"]
    status_counts = Counter()
    for p in per:
        if p.get("valid_strict"):
            status_counts["ok"] += 1
        else:
            for k in ("overlap", "dissoc", "desorbed", "intercalated", "surf_changed"):
                if p.get(f"has_{k}"):
                    status_counts[k] += 1
                    break

    # Attach compact flags without tensor-heavy records.
    rows = []
    for st, p in zip(pre_stats, per):
        q = dict(st)
        q.update({
            "valid_pre_relax": bool(p.get("valid_strict")),
            "pre_status": "ok" if p.get("valid_strict") else next(
                (k for k in ("overlap", "dissoc", "desorbed", "intercalated", "surf_changed") if p.get(f"has_{k}")),
                p.get("error") or "anomaly",
            ),
        })
        rows.append(q)

    def mean_key(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "ckpt": str(args.ckpt),
        "epoch": epoch,
        "n": len(rows),
        "elapsed_sec": time.time() - t0,
        "strict_pre_relax": strict["aggregate"],
        "pre_status_counts": dict(status_counts),
        "mae_movable_A": mean_key("mae_movable_A"),
        "mae_ads_A": mean_key("mae_ads_A"),
        "mae_surface_A": mean_key("mae_surface_A"),
        "ads_id_invalid_top": Counter(r["ads_id"] for r in rows if not r["valid_pre_relax"]).most_common(20),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
