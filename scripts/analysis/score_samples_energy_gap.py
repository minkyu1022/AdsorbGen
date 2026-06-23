#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adsorbgen.evaluation.energy import UMAEnergy  # noqa: E402


def _refs_from_records(raw) -> dict[str, float]:
    refs = {}
    items = raw.values() if isinstance(raw, dict) else raw
    for rec in items:
        if not isinstance(rec, dict):
            continue
        system_key = rec.get("system_id") or rec.get("system_key")
        energy = rec.get("e_sys_relaxed", rec.get("E_sys_min"))
        if system_key is not None and energy is not None:
            refs[str(system_key)] = float(energy)
    return refs


def _load_refs(cover_path: Path) -> dict[str, float]:
    if cover_path.is_file():
        if cover_path.suffix == ".pkl":
            with cover_path.open("rb") as f:
                return _refs_from_records(pickle.load(f))
        with cover_path.open() as f:
            return _refs_from_records(json.load(f))

    json_path = cover_path / "gt_results" / "global_minima.json"
    if json_path.exists():
        with json_path.open() as f:
            return _refs_from_records(json.load(f))

    pkl_path = cover_path / "oc20dense_mlip_global_min_by_system.pkl"
    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            return _refs_from_records(pickle.load(f))

    raise FileNotFoundError(
        f"No supported reference file under {cover_path}. Expected a .json/.pkl "
        "file, gt_results/global_minima.json, or oc20dense_mlip_global_min_by_system.pkl"
    )


def _as_single_pos(pos):
    if isinstance(pos, torch.Tensor) and pos.dim() == 3:
        return pos[0]
    return pos


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--cover-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--uma-model", default="uma-s-1p1")
    p.add_argument("--uma-task", default="oc20")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--success-margin", type=float, default=0.1)
    args = p.parse_args()

    payload = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = payload.get("records", payload)
    refs = _load_refs(Path(args.cover_dir))

    usable: list[tuple[dict, float]] = []
    missing = 0
    for rec in records:
        key = rec.get("system_key")
        ref = refs.get(str(key)) if key is not None else None
        if ref is None:
            missing += 1
            continue
        usable.append((rec, ref))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UMAEnergy(
        model_name=args.uma_model,
        task_name=args.uma_task,
        device=device,
        normalize_per_atom=False,
    )

    rows = []
    errors = 0
    chunk_size = max(int(args.batch_size), 1)
    for start in range(0, len(usable), chunk_size):
        chunk = usable[start:start + chunk_size]
        try:
            max_n = max(int(_as_single_pos(rec["pos_pred"]).shape[0]) for rec, _ in chunk)
            bsz = len(chunk)
            pos = torch.zeros((bsz, max_n, 3), dtype=torch.float32, device=device)
            cell = torch.zeros((bsz, 3, 3), dtype=torch.float32, device=device)
            atomic_numbers = torch.zeros((bsz, max_n), dtype=torch.long, device=device)
            pad_mask = torch.zeros((bsz, max_n), dtype=torch.bool, device=device)
            refs_t = torch.empty((bsz,), dtype=torch.float32, device=device)

            for j, (rec, ref) in enumerate(chunk):
                pos_j = _as_single_pos(rec["pos_pred"])
                n = int(pos_j.shape[0])
                pos[j, :n] = pos_j.to(device=device, dtype=torch.float32)
                cell[j] = rec["cell"].to(device=device, dtype=torch.float32)
                atomic_numbers[j, :n] = rec["atomic_numbers"].to(device=device, dtype=torch.long)
                pad_mask[j, :n] = True
                refs_t[j] = float(ref)

            e_pred = model(pos, cell, atomic_numbers, pad_mask)
            delta = (e_pred - refs_t).detach().cpu()
            e_pred_cpu = e_pred.detach().cpu()
            refs_cpu = refs_t.detach().cpu()
            for j, (rec, _) in enumerate(chunk):
                d = float(delta[j].item())
                if math.isfinite(d):
                    rows.append({
                        "sid": int(rec.get("sid", -1)),
                        "system_key": rec.get("system_key"),
                        "config_key": rec.get("config_key"),
                        "ads_id": int(rec.get("ads_id", -1)) if rec.get("ads_id") is not None else -1,
                        "e_pred": float(e_pred_cpu[j].item()),
                        "e_ref": float(refs_cpu[j].item()),
                        "delta": d,
                        "abs_delta": abs(d),
                        "success": d <= float(args.success_margin),
                    })
                else:
                    errors += 1
        except Exception as exc:
            errors += len(chunk)
            print(f"[energy] chunk {start}:{start + len(chunk)} failed: {exc}", flush=True)

        done = min(start + len(chunk), len(usable))
        print(f"[energy] {done}/{len(usable)} scored", flush=True)

    count = len(rows)
    delta_sum = sum(r["delta"] for r in rows)
    abs_sum = sum(r["abs_delta"] for r in rows)
    sq_sum = sum(r["delta"] * r["delta"] for r in rows)
    success_count = sum(1 for r in rows if r["success"])
    summary = {
        "samples": str(args.samples),
        "cover_dir": str(args.cover_dir),
        "uma_model": args.uma_model,
        "count": count,
        "missing": missing,
        "errors": errors,
        "mean_delta": delta_sum / count if count else None,
        "mean_abs_delta": abs_sum / count if count else None,
        "rmse_delta": math.sqrt(sq_sum / count) if count else None,
        "success_count": success_count,
        "success_rate": success_count / count if count else None,
        "success_margin": float(args.success_margin),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
