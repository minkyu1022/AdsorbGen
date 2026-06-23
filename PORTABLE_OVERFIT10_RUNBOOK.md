# Portable Overfit-10 Base/Recon Diagnostics

This package contains only the files needed to run two 10-system overfit
diagnostics on another server.

## What Is Included

- `adsorbgen/`: current AdsorbGen training/inference code.
- `geoopt/`: checkpoint loader used by the diagnostic scripts.
- `experiments/2026-06-overfit-denoiser-relax/`:
  - `train_x09_recon_denoiser_overfit.py`
  - `eval_x09_direct_recon_energy.py`
  - `eval_overfit_inference_relax_steps.py`
- `scripts/portable_overfit10/`:
  - `run_train_x1_base_overfit10.sh`
  - `run_train_x09_recon_denoiser_overfit10.sh`
  - `run_train_two_overfit10.sh`
  - `run_eval_x09_recon_energy.sh`
  - `run_eval_inference_relax_steps.sh`
- `data/overfit10/train.lmdb`: deterministic 10-system clean subset.
- `data/overfit10/selection.json`: source row metadata.
- `data/pkls/adsorbates.pkl`: fairchem adsorbate reference DB.

Full train LMDBs, W&B runs, checkpoints, cache dirs, and old replay artifacts
are intentionally excluded.

## Environment

Use the same AdsorbGen/fairchem environment used for normal training. At
minimum, the env must provide:

- PyTorch + Lightning
- fairchem-core with UMA pretrained MLIP access
- ASE, lmdb, numpy, scipy, pymatgen
- wandb only if online logging is wanted

From the unpacked directory:

```bash
cd AdsorbGen
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ADSORBATES_PKL="$PWD/data/pkls/adsorbates.pkl"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Train The Two Overfit Models

There are two training experiments on the same 10-system subset:

1. `overfit10_x1_base`: the normal base flow model overfit on 10 systems.
   Its diagnostic asks whether inference output still needs many post-relax
   LBFGS steps.
2. `overfit10_x09_recon_denoiser`: a direct denoiser trained only for
   `x1 + gamma(0.9) noise -> x1`.  Its diagnostic asks whether this direct
   reconstruction is energy-consistent after overfitting.

Sequential wrapper:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=AdsorbGen \
  bash scripts/portable_overfit10/run_train_two_overfit10.sh
```

To place the two runs on different visible GPU ids from separate shells, run
the individual commands below.

### Experiment 1: Plain x1 Base

Default command:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICES=1 BATCH_SIZE=64 EPOCHS=2000 \
  WANDB_PROJECT=AdsorbGen WANDB_RUN_NAME=overfit10_x1_base \
  bash scripts/portable_overfit10/run_train_x1_base_overfit10.sh
```

Important defaults:

- 10 physical LMDB rows are used.
- `--train-replicate 1000`, so each epoch exposes length 10000.
- `--val-replicate 100`, so validation sees repeated placement/time draws.
- `prediction_type=x1`, regular base coordinate loss.
- `gamma_schedule=none`; this is the plain base x1-prediction run.
- `ads_pair_l1_weight=1.0`.
- `prior_mode=random_heuristic`, `slab_source=initial`.

The output checkpoint defaults to:

```text
runs/overfit10_x1_base/last.ckpt
```

### Experiment 2: Direct x0.9 -> x1 Recon Denoiser

Default command:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICES=1 BATCH_SIZE=64 EPOCHS=2000 \
  bash scripts/portable_overfit10/run_train_x09_recon_denoiser_overfit10.sh
```

Important defaults:

- 10 physical LMDB rows are used.
- `--train-replicate 1000`, so each epoch exposes length 10000.
- It uses the same 102M `v0-ads-ref-adshead` backbone.
- It trains a direct x1 reconstruction target, not an auxiliary denoising head.
- The training input is `x_noisy = x1 + gamma(0.9) z` on movable atoms.
- The training target is direct `x1` prediction.
- `gamma_schedule=sqrt_t1mt`, `gamma_sigma=0.1`, `t=0.9`.
- `ads_pair_l1_weight=1.0`.
- `prior_mode=random_heuristic`, `slab_source=initial`.

The recon-denoiser output checkpoint defaults to:

```text
runs/overfit10_x09_recon_denoiser/last.ckpt
```

To start from a base checkpoint, append:

```bash
EXTRA_ARGS='--init-from-ckpt /path/to/base.ckpt'
```

or edit the train script and add `--init-from-ckpt`.

## Diagnostic 1: x1 -> x0.9 Noise -> Direct Recon Energy

This diagnostic requires the direct recon-denoiser checkpoint.

Direct endpoint-noise diagnostic:

```bash
CKPT=runs/overfit10_x09_recon_denoiser/last.ckpt \
  OUT=runs/overfit10_x09_direct_recon_energy \
  T_VALUE=0.9 \
  bash scripts/portable_overfit10/run_eval_x09_recon_energy.sh
```

This forms:

```text
x_noisy = x1 + gamma(0.9) z
x1_recon = f_theta(x0_context, x_noisy, t=0.9)
```

It writes:

- `summary.json`
- `rows.json`

Key fields:

- `recon_delta_E_mae_eV`
- `recon_delta_E_max_abs_eV`
- `rmsd_recon_to_x1_mean_A`
- `noisy_delta_E_mae_eV`

## Diagnostic 2: Inference Output -> Post Relaxation Steps

Run this diagnostic for the base overfit checkpoint and compare `model_pred`
against `random_prior` and `target_relaxed`.

```bash
CKPT=runs/overfit10_x1_base/last.ckpt \
  OUT=runs/overfit10_base_inference_relax_steps \
  FLOW_STEPS=50 FMAX=0.05 MAX_STEPS=300 \
  bash scripts/portable_overfit10/run_eval_inference_relax_steps.sh
```

For each of the same 10 rows, it relaxes three starts:

- `model_pred`: flow inference output.
- `random_prior`: the random heuristic x0 prior.
- `target_relaxed`: LMDB MLIP-relaxed target position.

The wrapper defaults are ASE-LBFGS-like:

- `UMA_MODEL=uma-s-1p1`
- `FMAX=0.05`
- `MAX_STEPS=300`
- `LBFGS_MAXSTEP=0.2`
- `LBFGS_MEMORY=100`

For the older replay/custom-batched setting, explicitly override
`LBFGS_MAXSTEP=0.04 LBFGS_MEMORY=50 UMA_MODEL=uma-s-1p2`. Step counts from that
setting should not be compared directly with ASE-default step counts.

It writes:

- `summary.json`: mean convergence rate, step count, and energy drop by start.
- `rows.json`: per-system values.

Interpretation:

- If `model_pred` has near-zero or much lower `n_steps_mean` than
  `random_prior`, the overfit model is putting samples close to the UMA basin.
- If `model_pred` still needs many steps while `target_relaxed` is near zero,
  the model output is not reproducing the MLIP-relaxed target geometry.
- If `target_relaxed` itself needs many steps, the LMDB target and current UMA
  relaxation criterion/model are not exactly identical.
