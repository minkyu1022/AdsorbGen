"""End-to-end one-variant runner for AdsorbGen architecture search.

Trains a named variant for a short budget, runs inference on the validation
LMDB, runs the geometric evaluator, and writes a compact metrics row to
``runs/<variant>/search_metrics.json`` so ``search_rank.py`` can aggregate.

Each variant gets its own ``runs/<variant>/`` directory which auto-resumes
on re-run (via train.py's existing ``last.ckpt`` logic), so the search loop
can be interrupted and restarted without losing progress.

Intentionally a thin wrapper around ``adsorbgen.train``, ``adsorbgen.inference``,
and ``adsorbgen.eval`` — the goal is orchestration, not reinvention. All three
sub-steps run as separate subprocesses to keep their module-level state
isolated (Lightning, fairchem).

Usage:

    PYTHONPATH=AdsorbGen python -m adsorbgen.scripts.search_run \
        --variant v3-no-mic-dist \
        --train-lmdb data/processed/is2res_train.lmdb \
                     data/processed/is2res_val_ood_ads.lmdb \
                     data/processed/is2res_val_ood_cat.lmdb \
                     data/processed/is2res_val_ood_both.lmdb \
        --val-lmdb   data/processed/oc20dense.lmdb \
        --runs-root  runs \
        --epochs 20 --batch-size 32 --devices 4 \
        --eval-max-samples 256

The final JSON row looks like::

    {
      "variant": "v3-no-mic-dist",
      "epochs": 20,
      "n_samples": 256,
      "displacement_mae_A": 0.1234,
      "valid_rate_strict": 0.812,
      "overlap_rate": 0.150,
      "dissoc_rate": 0.038,
      "ckpt": "runs/v3-no-mic-dist/last.ckpt"
    }
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], log_path: Path, env: dict | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[search_run] $ {' '.join(cmd)} (log: {log_path})", flush=True)
    with open(log_path, "a") as f:
        f.write(f"\n\n===== {' '.join(cmd)} =====\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if rc != 0:
        raise SystemExit(f"[search_run] step failed (rc={rc}): {' '.join(cmd)}\n  see {log_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--arch", choices=["v1", "v2"], default="v2",
                   help="passes through to adsorbgen.train; v1 ignores --variant overrides")
    p.add_argument("--train-lmdb", nargs="+", required=True)
    p.add_argument("--val-lmdb", required=True)
    p.add_argument("--runs-root", default="runs")

    # Training budget (small on purpose — search loop signal, not final perf).
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-warmup-steps", type=int, default=500)
    p.add_argument("--wandb-project", type=str, default=None)

    # Eval budget.
    p.add_argument("--eval-max-samples", type=int, default=256)
    p.add_argument("--eval-num-steps", type=int, default=20)
    p.add_argument("--eval-batch-size", type=int, default=16)

    p.add_argument("--python", default=sys.executable)
    p.add_argument("--cuda-visible-devices", default=None)
    args = p.parse_args()

    out_dir = Path(args.runs_root) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "search_run.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = "AdsorbGen" + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # -------- 1. Train ---------------------------------------------------
    train_cmd = [
        args.python, "-m", "adsorbgen.train",
        "--train-lmdb", *args.train_lmdb,
        "--val-lmdb", args.val_lmdb,
        "--out", str(out_dir),
        "--arch", args.arch,
        "--variant", args.variant,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--devices", str(args.devices),
        "--lr", str(args.lr),
        "--lr-warmup-steps", str(args.lr_warmup_steps),
        "--loss-type", "l1",
        "--activation-checkpointing",
    ]
    if args.wandb_project:
        train_cmd += ["--wandb-project", args.wandb_project, "--wandb-run-name", args.variant]
    _run(train_cmd, log_path, env=env)

    ckpt = out_dir / "last.ckpt"
    if not ckpt.exists():
        raise SystemExit(f"[search_run] expected {ckpt} after training, not found")

    # -------- 2. Inference (single-placement, K=1, no FK) ---------------
    samples_path = out_dir / "search_samples.pt"
    infer_cmd = [
        args.python, "-m", "adsorbgen.inference",
        "--ckpt", str(ckpt),
        "--lmdb", args.val_lmdb,
        "--out", str(samples_path),
        "--batch-size", str(args.eval_batch_size),
        "--num-workers", "0",
        "--num-steps", str(args.eval_num_steps),
        "--max-samples", str(args.eval_max_samples),
    ]
    _run(infer_cmd, log_path, env=env)

    # -------- 3. Eval ----------------------------------------------------
    metrics_path = out_dir / "search_metrics_full.json"
    eval_cmd = [
        args.python, "-m", "adsorbgen.eval",
        "--samples", str(samples_path),
        "--out", str(metrics_path),
    ]
    _run(eval_cmd, log_path, env=env)

    # -------- 4. Row summary --------------------------------------------
    with open(metrics_path) as f:
        full = json.load(f)
    agg = full["aggregate"]
    row = {
        "variant": args.variant,
        "epochs": args.epochs,
        "n_samples": agg["n_samples"],
        "displacement_mae_A": agg["displacement_mae_A"],
        "valid_rate_strict": agg["valid_rate_strict"],
        "overlap_rate": agg["overlap_rate"],
        "dissoc_rate": agg["dissoc_rate"],
        "ckpt": str(ckpt),
    }
    row_path = out_dir / "search_metrics.json"
    with open(row_path, "w") as f:
        json.dump(row, f, indent=2)
    print(
        f"[search_run] {args.variant}: "
        f"mae={row['displacement_mae_A']:.4f} valid_strict={row['valid_rate_strict']:.3f} "
        f"(n={row['n_samples']}) -> {row_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
