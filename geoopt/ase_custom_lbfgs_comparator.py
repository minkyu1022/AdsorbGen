#!/usr/bin/env python
"""Compare production ASE LBFGS and custom batched LBFGS on identical jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import LBFGS

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geoopt import atomic_write_json, build_relax_jobs, load_uma, run_optimizer  # noqa: E402
from adsorbgen.evaluation.metrics import load_pristine_context, _score_record_anomaly  # noqa: E402


def fixed_atoms_from_prediction(p: dict) -> Atoms:
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


def relax_ase_one(job: dict, calc, args) -> dict:
    atoms = fixed_atoms_from_prediction(job["relax_input"])
    atoms.calc = calc
    opt = LBFGS(
        atoms,
        logfile=None,
        maxstep=float(args.maxstep),
        memory=int(args.lbfgs_memory),
        damping=float(args.lbfgs_damping),
        alpha=float(args.lbfgs_alpha),
    )
    try:
        converged = bool(opt.run(fmax=float(args.fmax), steps=int(args.max_steps)))
        e_sys = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        fmax = float(np.max(np.linalg.norm(forces, axis=1)))
        relaxed_pos = atoms.get_positions().astype(np.float32)
        err = None
    except Exception as exc:
        converged = False
        e_sys = float("nan")
        fmax = float("nan")
        relaxed_pos = np.asarray(job["relax_input"]["pos_pred"], dtype=np.float32)
        err = repr(exc)
    return {
        "global_i": int(job["global_i"]),
        "n_atoms": int(len(job["relax_input"]["numbers"])),
        "converged": bool(converged),
        "E_sys": float(e_sys),
        "fmax": float(fmax),
        "n_steps": int(getattr(opt, "nsteps", 0)),
        "pos_relaxed": relaxed_pos,
        "error": err,
    }


def relax_ase_batch(jobs: list[dict], args, device: torch.device) -> list[dict]:
    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    predict_unit = pretrained_mlip.get_predict_unit(args.uma_model, device=str(device))
    if not args.use_fairchem_batcher:
        calc = FAIRChemCalculator(predict_unit, task_name=args.uma_task)
        return [relax_ase_one(job, calc, args) for job in jobs]

    from fairchem.core.calculate._batch import InferenceBatcher

    batcher = InferenceBatcher(
        predict_unit,
        max_batch_size=int(args.batcher_max_atoms),
        batch_wait_timeout_s=float(args.batcher_wait_timeout_s),
        concurrency_backend_options={"max_workers": int(args.lbfgs_concurrency)},
    )
    tls = threading.local()

    def thread_calc():
        local_calc = getattr(tls, "calc", None)
        if local_calc is None:
            local_calc = FAIRChemCalculator(batcher.batch_predict_unit, task_name=args.uma_task)
            tls.calc = local_calc
        return local_calc

    def run_one(i: int, job: dict) -> tuple[int, dict]:
        return i, relax_ase_one(job, thread_calc(), args)

    out: list[dict | None] = [None] * len(jobs)
    futures = [batcher.executor.submit(run_one, i, job) for i, job in enumerate(jobs)]
    for fut in as_completed(futures):
        i, result = fut.result()
        out[i] = result
    return [r for r in out if r is not None]


def valid_status(job: dict, result: dict) -> tuple[bool, str | None]:
    if not result["converged"] or not np.isfinite(float(result["E_sys"])):
        return False, "unconverged"
    p = job["relax_input"]
    ar = _score_record_anomaly(
        {
            "sid": int(job["sid"]),
            "system_key": tuple(job["system_key"]),
            "ads_id": int(job.get("ads_id", -1)),
            "pos_ref": torch.as_tensor(job["pos_ref"], dtype=torch.float32),
            "pos_pred": torch.as_tensor(result["pos_relaxed"], dtype=torch.float32),
            "pos_gt": torch.as_tensor(job["pos_gt"], dtype=torch.float32),
            "atomic_numbers": torch.as_tensor(p["numbers"], dtype=torch.long),
            "tags": torch.as_tensor(p["tags"], dtype=torch.long),
            "cell": torch.as_tensor(p["cell"], dtype=torch.float32),
        }
    )
    if ar.get("valid_strict"):
        return True, None
    flags = [
        k for k in ("overlap", "dissoc", "desorbed", "intercalated", "surf_changed")
        if ar.get(f"has_{k}")
    ]
    return False, flags[0] if flags else ar.get("error") or "anomaly"


def annotate_decisions(jobs: list[dict], results: list[dict], window_ev: float) -> list[dict]:
    by_job = {int(job["global_i"]): job for job in jobs}
    out = []
    for r in results:
        job = by_job[int(r["global_i"])]
        valid, anomaly = valid_status(job, r)
        e_ref = float(job["E_sys_ref"])
        e_sys = float(r["E_sys"])
        out.append(
            {
                **{k: v for k, v in r.items() if k != "pos_relaxed"},
                "system_i": int(job["system_i"]),
                "sample_i": int(job["sample_i"]),
                "sid": int(job["sid"]),
                "system_key": list(job["system_key"]),
                "E_sys_ref": e_ref,
                "improvement": float(e_ref - e_sys) if np.isfinite(e_sys) else float("nan"),
                "valid": bool(valid),
                "anomaly": anomaly,
                "success": bool(valid and r["converged"] and np.isfinite(e_sys) and e_sys < e_ref),
                "window": bool(valid and r["converged"] and np.isfinite(e_sys) and e_sys <= e_ref + float(window_ev)),
            }
        )
    return out


def finite_abs(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.abs(arr[np.isfinite(arr)])


def compare(ase_rows: list[dict], custom_rows: list[dict]) -> dict:
    def row_key(r: dict) -> tuple[int, int]:
        return (int(r["sid"]), int(r.get("sample_i", 0)))

    custom_by_id = {row_key(r): r for r in custom_rows}
    rows = []
    for a in ase_rows:
        c = custom_by_id[row_key(a)]
        d_e = float(c["E_sys"] - a["E_sys"])
        rows.append(
            {
                "global_i": int(a["global_i"]),
                "sid": int(a["sid"]),
                "ase_converged": bool(a["converged"]),
                "custom_converged": bool(c["converged"]),
                "ase_valid": bool(a["valid"]),
                "custom_valid": bool(c["valid"]),
                "ase_success": bool(a["success"]),
                "custom_success": bool(c["success"]),
                "ase_window": bool(a["window"]),
                "custom_window": bool(c["window"]),
                "ase_E": float(a["E_sys"]),
                "custom_E": float(c["E_sys"]),
                "dE": d_e,
                "ase_fmax": float(a["fmax"]),
                "custom_fmax": float(c["fmax"]),
                "ase_steps": int(a["n_steps"]),
                "custom_steps": int(c["n_steps"]),
                "ase_anomaly": a["anomaly"],
                "custom_anomaly": c["anomaly"],
            }
        )
    finite_de = finite_abs([r["dE"] for r in rows])
    both_conv = [r for r in rows if r["ase_converged"] and r["custom_converged"] and np.isfinite(r["dE"])]
    both_conv_abs = finite_abs([r["dE"] for r in both_conv])

    def pct(arr: np.ndarray, q: float) -> float | None:
        return float(np.percentile(arr, q)) if arr.size else None

    summary = {
        "n": len(rows),
        "ase_converged": sum(r["ase_converged"] for r in rows),
        "custom_converged": sum(r["custom_converged"] for r in rows),
        "converged_agreement": sum(r["ase_converged"] == r["custom_converged"] for r in rows),
        "ase_converged_custom_not": sum(r["ase_converged"] and not r["custom_converged"] for r in rows),
        "custom_converged_ase_not": sum(r["custom_converged"] and not r["ase_converged"] for r in rows),
        "valid_agreement": sum(r["ase_valid"] == r["custom_valid"] for r in rows),
        "success_flips": sum(r["ase_success"] != r["custom_success"] for r in rows),
        "window_flips": sum(r["ase_window"] != r["custom_window"] for r in rows),
        "both_converged": len(both_conv),
        "abs_dE_median": pct(finite_de, 50),
        "abs_dE_p95": pct(finite_de, 95),
        "abs_dE_max": float(np.max(finite_de)) if finite_de.size else None,
        "both_converged_abs_dE_median": pct(both_conv_abs, 50),
        "both_converged_abs_dE_p95": pct(both_conv_abs, 95),
        "both_converged_abs_dE_max": float(np.max(both_conv_abs)) if both_conv_abs.size else None,
    }
    return {"summary": summary, "rows": rows}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/irteam/AdsorbGen")
    ap.add_argument("--adsorbates-pkl", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-lmdb", nargs="+", required=True)
    ap.add_argument("--selected-systems", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--seed", type=int, default=20260526)
    ap.add_argument("--num-systems", type=int, default=100)
    ap.add_argument("--system-offset", type=int, default=0)
    ap.add_argument("--num-placements", type=int, default=1)
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--flow-batch-size", type=int, default=32)
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--uma-model", default="uma-s-1p2")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-atoms", type=int, default=4096)
    ap.add_argument("--maxstep", type=float, default=0.04)
    ap.add_argument("--lbfgs-memory", type=int, default=50)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    ap.add_argument("--lbfgs-history-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-position-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-curvature-guard", choices=["abs", "positive", "ase"], default="abs")
    ap.add_argument("--window-ev", type=float, default=0.1)
    ap.add_argument("--pristine-slabs", default="")
    ap.add_argument("--pristine-index", default="")
    ap.add_argument("--use-fairchem-batcher", action="store_true")
    ap.add_argument("--lbfgs-concurrency", type=int, default=1)
    ap.add_argument("--batcher-max-atoms", type=int, default=4096)
    ap.add_argument("--batcher-wait-timeout-s", type=float, default=0.02)
    ap.add_argument(
        "--ase-reference-json",
        nargs="*",
        default=[],
        help="existing comparator JSON files; when set, reuse their ase_rows and skip ASE relaxation",
    )
    return ap


def load_reference_ase_rows(paths: list[str], jobs: list[dict]) -> list[dict]:
    wanted = {(int(job["sid"]), int(job["sample_i"])) for job in jobs}
    rows = []
    seen = set()
    for raw_path in paths:
        with Path(raw_path).open() as f:
            doc = json.load(f)
        for row in doc.get("ase_rows", []):
            key = (int(row["sid"]), int(row.get("sample_i", 0)))
            if key in wanted and key not in seen:
                rows.append(row)
                seen.add(key)
    missing = wanted - seen
    if missing:
        raise RuntimeError(f"ASE reference rows missing for {len(missing)} jobs")
    return rows


def main() -> None:
    args = parser().parse_args()
    os.environ["ADSGEN_ROOT"] = str(Path(args.repo).resolve())
    os.environ["ADSORBATES_PKL"] = str(Path(args.adsorbates_pkl).resolve())
    if args.pristine_slabs:
        load_pristine_context(Path(args.pristine_slabs), Path(args.pristine_index) if args.pristine_index else None)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")

    t0 = time.time()
    jobs = build_relax_jobs(args, device)
    flow_elapsed = time.time() - t0

    if args.ase_reference_json:
        t_ase = time.time()
        ase_rows = load_reference_ase_rows(args.ase_reference_json, jobs)
        ase_elapsed = time.time() - t_ase
    else:
        t_ase = time.time()
        ase_results = relax_ase_batch(jobs, args, device)
        ase_elapsed = time.time() - t_ase
        ase_rows = annotate_decisions(jobs, ase_results, args.window_ev)

    uma = load_uma(args.uma_model, args.uma_task, device)
    t_custom = time.time()
    custom_results = run_optimizer(jobs, uma, args, device, "lbfgs", serial=False)
    custom_elapsed = time.time() - t_custom
    custom_rows = annotate_decisions(jobs, custom_results, args.window_ev)

    cmp = compare(ase_rows, custom_rows)
    report: dict[str, Any] = {
        "settings": vars(args),
        "timing": {
            "flow_elapsed_sec": flow_elapsed,
            "ase_elapsed_sec": ase_elapsed,
            "custom_elapsed_sec": custom_elapsed,
            "ase_candidates_per_sec": len(jobs) / ase_elapsed if ase_elapsed > 0 else None,
            "custom_candidates_per_sec": len(jobs) / custom_elapsed if custom_elapsed > 0 else None,
        },
        "summary": cmp["summary"],
        "rows": cmp["rows"],
        "ase_rows": ase_rows,
        "custom_rows": custom_rows,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, report)
    print(json.dumps({"timing": report["timing"], "summary": report["summary"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
