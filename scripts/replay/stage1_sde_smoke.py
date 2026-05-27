"""Stage 1: pre-UMA valid rate smoke test for AtomMOF SDE inference.

For each system in an LMDB, samples K placements via the configured prior,
runs euler_sample (ODE or AtomMOF SDE), and checks the 5-axis anomaly on
the predicted x_1 (no UMA relaxation). Reports per-shard aggregate.

CLI:
  python stage1_sde_smoke.py --ckpt PATH --lmdb PATH --out PATH
      --use-sde --num-placements 5 --shard-idx 0 --num-shards 4
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
import torch
from tqdm.auto import tqdm

import adsorbgen.models.dit as _dit_mod
import adsorbgen.models.dit_v2 as _dit_v2_mod
sys.modules.setdefault("adsorbgen.model", _dit_mod)
sys.modules.setdefault("adsorbgen.model.dit", _dit_mod)
sys.modules.setdefault("adsorbgen.model.dit_v2", _dit_v2_mod)

from adsorbgen.flow import FlowConfig, euler_sample
from adsorbgen.models.dit import DiTDenoiserConfig
from adsorbgen.models.dit_v2 import DiTDenoiserV2Config
from adsorbgen.models.factory import build_model
from adsorbgen.data.multiplace import DEFAULT_ADSORBATES_PKL
from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement
from adsorbgen.replay.eval import _passes_anomaly


def _filter_fields(cls, payload: dict) -> dict:
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    return {k: v for k, v in payload.items() if k in valid}


def _resolve_model_cfg(args_json_path: Path):
    a = json.load(open(args_json_path))
    arch = a.get("arch", "v1")
    payload = a.get("model_config", a)
    if arch == "v1":
        return DiTDenoiserConfig(**_filter_fields(DiTDenoiserConfig, payload))
    return DiTDenoiserV2Config(**_filter_fields(DiTDenoiserV2Config, payload))


def _extract_state_dict(state):
    if "state_dict" in state:
        sd = state["state_dict"]
        return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    return state.get("model", state)


def _load_model(ckpt_path: Path, device: torch.device):
    torch.serialization.add_safe_globals([DiTDenoiserConfig, DiTDenoiserV2Config, FlowConfig])
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = _extract_state_dict(ck)
    cfg = _resolve_model_cfg(ckpt_path.parent / "args.json")
    model = build_model(cfg)
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-placements", type=int, default=5)
    ap.add_argument("--prior-mode", default="catflow_center_rel")
    ap.add_argument("--use-sde", action="store_true")
    ap.add_argument("--sde-schedule", type=str, default="atommof",
                    choices=["atommof", "zero_ends"])
    ap.add_argument("--sde-alpha", type=float, default=1.0)
    ap.add_argument("--sde-no-score", action="store_true")
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--max-systems", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + args.shard_idx)
    np.random.seed(args.seed + args.shard_idx)

    model, model_cfg = _load_model(Path(args.ckpt), device)
    flow_cfg = FlowConfig(eps=1e-5, prediction_type="x1")
    use_ads_ref = bool(getattr(model_cfg, "use_ads_ref_pos", False))

    # Use PlacementPriorDataset which supports catflow_center_rel etc.
    # (MultiPlacementDataset only supports fairchem random/heuristic priors.)
    base_ds = PlacementPriorDataset(
        lmdb_path=args.lmdb,
        prior_mode=args.prior_mode,
        adsorbates_pkl=DEFAULT_ADSORBATES_PKL,
        recenter=True,
        training_aug=False,
        provide_ads_ref_pos=use_ads_ref,
    )

    # Build representative-per-unique-system index (first clean entry per system_key).
    import lmdb as _lmdb
    env = _lmdb.open(args.lmdb, subdir=False, readonly=True, lock=False)
    unique_base_indices = []
    seen = set()
    with env.begin() as txn:
        n = txn.stat()["entries"]
        for i in range(n):
            raw = txn.get(str(i).encode("ascii"))
            if raw is None:
                continue
            e = pickle.loads(raw)
            if int(e.get("anomaly", 0)) != 0:
                continue
            sk = e.get("system_key")
            if sk is None or sk in seen:
                continue
            seen.add(sk)
            unique_base_indices.append(i)
    env.close()

    if args.max_systems:
        unique_base_indices = unique_base_indices[:args.max_systems]

    indices = [bi for j, bi in enumerate(unique_base_indices) if j % args.num_shards == args.shard_idx]
    print(f"[shard{args.shard_idx}] {len(indices)}/{len(unique_base_indices)} unique systems × "
          f"{args.num_placements} placements = {len(indices) * args.num_placements} candidates",
          flush=True)

    # Build candidate task list: (base_i, placement_i). Each (base_i, k) draws
    # a fresh placement (catflow Gaussian) from the underlying dataset with a
    # deterministic seed so reruns are reproducible.
    tasks = [(bi, k) for bi in indices for k in range(args.num_placements)]

    def _draw_sample(bi: int, k: int) -> dict:
        seed = (args.seed * 1_000_003 + bi * 23 + k) & 0xFFFF_FFFF
        np.random.seed(seed)
        return base_ds[bi]

    rows = []
    t0 = time.time()
    for start in tqdm(range(0, len(tasks), args.batch_size),
                       desc=f"[shard{args.shard_idx}] sample+anomaly",
                       unit="batch", dynamic_ncols=True):
        chunk = tasks[start:start + args.batch_size]
        samples = [_draw_sample(bi, k) for (bi, k) in chunk]
        batch = collate_displacement(samples)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Movable mask follows ckpt training (use model's _runtime_movable_mask logic)
        # For surface_ads, movable = (tags != 0) & pad_mask  (sub-surface + ads)
        # For adsorbate_only, movable = (tags == 2) & pad_mask
        movable_mode = str(getattr(model, "adsorbgen_movable_mode", "surface_ads"))
        if movable_mode == "adsorbate_only":
            movable = (batch["tags"] == 2) & batch["pad_mask"]
        else:
            movable = (batch["tags"] >= 1) & batch["pad_mask"]

        def fwd(x_t, t, _b=batch):
            extra = {"ads_ref_pos": _b["ads_ref_pos"]} if use_ads_ref else {}
            return model(
                pos=_b["pos"], x_t=x_t, t=t,
                atomic_numbers=_b["atomic_numbers"], tags=_b["tags"],
                movable_mask=movable, pad_mask=_b["pad_mask"], cell=_b["cell"],
                **extra,
            )

        x_out = euler_sample(
            fwd, batch["pos"], movable, batch["pad_mask"], flow_cfg,
            num_steps=args.num_steps,
            use_sde=args.use_sde,
            refine_final=False,
            sde_schedule=args.sde_schedule,
            sde_alpha=args.sde_alpha,
            sde_no_score=args.sde_no_score,
        )

        # Anomaly check per candidate
        for i, (bi, k) in enumerate(chunk):
            n = int(batch["pad_mask"][i].sum().item())
            tags = batch["tags"][i, :n].detach().cpu().numpy().astype(np.int64)
            numbers = batch["atomic_numbers"][i, :n].detach().cpu().numpy().astype(np.int64)
            cell = batch["cell"][i].detach().cpu().numpy()
            if cell.ndim == 3:
                cell = cell[0]
            pos_ref = batch["pos"][i, :n].detach().cpu().numpy().astype(np.float64)
            pos_pred = x_out[i, :n].detach().cpu().numpy().astype(np.float64)
            pos_gt = batch["pos_relaxed"][i, :n].detach().cpu().numpy().astype(np.float64)
            sid = int(batch["sid"][i].item())
            ads_id = int(batch["ads_id"][i].item()) if "ads_id" in batch else -1

            passed, reason = _passes_anomaly(
                pos_ref=pos_ref, pos_pred=pos_pred, pos_gt=pos_gt,
                atomic_numbers=numbers, tags=tags, cell=cell.astype(np.float32),
                sid=sid, ads_id=ads_id,
            )
            rows.append({
                "sid": sid, "base_i": int(bi), "placement_i": int(k),
                "valid": bool(passed), "reason": reason,
            })

    # Aggregate
    total = len(rows)
    valid = sum(1 for r in rows if r["valid"])
    by_reason = {}
    for r in rows:
        if not r["valid"]:
            by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1

    summary = {
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
        "ckpt": str(Path(args.ckpt).resolve()),
        "lmdb": str(Path(args.lmdb).resolve()),
        "use_sde": args.use_sde,
        "sde_schedule": args.sde_schedule,
        "sde_alpha": args.sde_alpha,
        "sde_no_score": args.sde_no_score,
        "num_placements": args.num_placements,
        "num_steps": args.num_steps,
        "elapsed_sec": time.time() - t0,
        "total_candidates": total,
        "valid": valid,
        "valid_rate": valid / max(total, 1),
        "anomaly_breakdown": by_reason,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_path.with_suffix(".rows.pkl"), "wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[shard{args.shard_idx}] valid_rate = {valid/max(total,1)*100:.1f}% ({valid}/{total})", flush=True)


if __name__ == "__main__":
    main()
