# AdsorbGen Server Resume Memo

This memo is for moving the current AdsorbGen pipeline from
`/home/minkyu/Cat-bench` to another server. A new agent should be able to read
this file once and know what code, data, and external packages are required.

## Current Direction

- Continue the DiT-based AdsorbGen line.
- Do not continue the pure typed GNN experiment; it failed and should not be
  treated as a required path.
- Main successful direction so far:
  - `arch=v1`
  - `variant=v0-ads-ref` or `v0-ads-ref-adshead`
  - `prediction_type=x1`
  - L1 coordinate loss
  - auxiliary adsorbate-internal all-pair distance L1 loss
  - `ADS_PAIR_L1_WEIGHT=1.0`
  - no bond loss and no nonbonded clash loss for the best all-pair baseline

## Code Repositories

Primary repo:

```text
https://github.com/minkyu1022/AdsorbGen.git
local: /home/minkyu/Cat-bench/AdsorbGen
```

External dependency repo for fast batched UMA relaxation:

```text
https://github.com/minkyu1022/fast_dynamics.git
local: /home/minkyu/fast_dynamics
```

On the new server:

```bash
git clone https://github.com/minkyu1022/AdsorbGen.git
git clone https://github.com/minkyu1022/fast_dynamics.git
pip install -r AdsorbGen/requirements-server.txt
pip install -e ./fast_dynamics
```

`fast_dynamics` depends on `torch`, `ase`, `fairchem-core`, and
`nvalchemi-toolkit`. Replay uses:

```python
from fast_dynamics import UMAWrapper, prepare_batch_for_dynamics
from nvalchemi.dynamics import FIRE, ConvergenceHook
```

Fast dynamics correctness was previously smoke-tested against FAIRChem UMA:
bridge tests passed, real GPU UMA wrapper tests passed, and fixed atoms stayed
fixed under the registered hook.

Important path defaults:

- `ADSORBATES_PKL` can override the adsorbate reference DB path.
- Without the env var, AdsorbGen expects:
  `data/pkls/adsorbates.pkl` under the Cat-bench project root.
- `CAT_BENCH_ROOT` can override the project root for replay defaults.
- Without the env var, replay defaults resolve relative to the AdsorbGen
  checkout's parent directory.

## Code Cleanup Before Push

The failed pure typed-GNN experiment is intentionally not part of the server
handoff. Keep the DiT/all-pair line only; do not upload or reintroduce:

```text
adsorbgen/model_typed_gnn.py
docs/typedgnn_implementation_memo.md
--arch typedgnn / variant typedgnn paths
```

Keep the DiT-related changes:

```text
adsorbgen/dataset.py
adsorbgen/energy.py
adsorbgen/eval.py
adsorbgen/eval_replay.py
adsorbgen/flow.py
adsorbgen/inference.py
adsorbgen/model.py
adsorbgen/model_factory.py
adsorbgen/model_v2.py
adsorbgen/multiplace.py
adsorbgen/replay.py
adsorbgen/replay_viz.py
adsorbgen/train.py
adsorbgen/transformer.py
adsorbgen/variants.py
adsorbgen/tests/test_eval.py
adsorbgen/tests/test_lddt_loss.py
```

Also move root-level operational scripts into the repo or keep them beside the
repo on the new server:

```text
run_refine.sh
run_including_anomaly.sh
run_w_ads_ref_pos_no_replay.sh
scripts/revalidate_pristine.py
scripts/evaluate_multisample_validity.py
scripts/diagnose_overlap_pairs.py
scripts/replay_one_ckpt.py
scripts/run_replay_4gpu.sh
scripts/merge_replay_shards.py
scripts/run_replay_probe.py
scripts/build_replay_gt_index.py
scripts/extract_pristine_slabs.py
scripts/compute_gas_refs.py
scripts/preflight_placement.py
viz/
```

Keep `viz/frontend/package-lock.json` with `viz/frontend/package.json`.
Do not upload/commit `node_modules` or `.next`.

## Processed Dataset And Anomaly Handling

The processed LMDBs already contain both clean and anomaly samples. They are
not physically removed during preprocessing.

Each processed entry contains an `anomaly` field, and each LMDB has an
`anomaly_mask` metadata key. Loader behavior decides whether to use or skip
anomaly entries:

- `skip_anomaly=True`: use only entries where `anomaly_mask == 0`.
- `skip_anomaly=False`: use all entries.
- `run_including_anomaly.sh` passes `--include-anomaly`, which makes training
  use `skip_anomaly=False`.
- Validation/revalidation scripts generally keep `skip_anomaly=True` for clean
  comparison.

Current processed LMDB counts:

| LMDB | total | clean | anomaly |
| --- | ---: | ---: | ---: |
| `is2res_train.lmdb` | 460,328 | 345,254 | 115,074 |
| `is2res_val.lmdb` | 24,943 | 18,683 | 6,260 |
| `is2res_val_ood_ads.lmdb` | 24,961 | 19,485 | 5,476 |
| `is2res_val_ood_cat.lmdb` | 24,963 | 18,821 | 6,142 |
| `is2res_val_ood_both.lmdb` | 24,987 | 20,901 | 4,086 |
| `oc20dense.lmdb` | 65,073 | 65,073 | 0 |

Conclusion: do not rerun preprocessing just to include anomaly samples. The
all-sample processed form already exists. Use `--include-anomaly` when training
on anomaly entries is desired.

## Canonical Data Names

Use one dense filename everywhere:

```text
data/processed/oc20dense.lmdb
```

Do not create or rely on `oc20dense_test.lmdb`.

## Required Dropbox Upload Bundle

Upload this file (`for_resum.md`) together with the data bundle. The canonical
upload manifest is `dropbox_upload_manifest.txt` in the Cat-bench root and is
also copied into the repo as `AdsorbGen/docs/dropbox_upload_manifest.txt`.

Do not upload whole local run/log directories. In particular, local `runs/` is
hundreds of GB and mostly contains obsolete checkpoints, launch logs,
revalidation scratch outputs, replay visualization artifacts, and failed
ablation runs. Upload only the single all-pair ep29 checkpoint listed below.

Required processed LMDBs:

```text
data/processed/is2res_train.lmdb
data/processed/is2res_val.lmdb
data/processed/is2res_val_ood_ads.lmdb
data/processed/is2res_val_ood_cat.lmdb
data/processed/is2res_val_ood_both.lmdb
data/processed/oc20dense.lmdb
```

Required adsorbate reference DB:

```text
data/pkls/adsorbates.pkl
```

This file is small but mandatory for active `use_ads_ref_pos=True` runs. It is
used for `ads_ref_pos` lookup during training/inference placement.

Required pristine slab references:

```text
results/pristine_slabs/is2res.pkl
results/pristine_slabs/is2res.sid_index.pkl
results/pristine_slabs/oc20dense_uma.pkl
results/pristine_slabs/oc20dense.system_index.pkl
```

`oc20dense_uma.pkl` is the OC20-Dense clean-slab reference produced by:

1. extracting final clean-surface frames from
   `data/oc20dense/oc20_dense_trajectories.tar.gz`, keyed by
   `data/oc20dense/oc20dense_mapping.pkl`;
2. relaxing those unique slabs with UMA on GPUs 4-7 while freezing bulk/fixed
   atoms according to `data/processed/oc20dense.lmdb`.

The extraction input `results/pristine_slabs/oc20dense.pkl` is useful for audit,
but the upload/reference target for Dense surface-change evaluation should be
the UMA-relaxed `oc20dense_uma.pkl`. Dense lookup uses `system_key` through
`oc20dense.system_index.pkl`; it cannot use `sid` because Dense entries have
`sid = -1`.

Required pre-relaxed structure/energy artifacts:

```text
results/gas_phase_refs.pkl
results/phase1/bulk_results.pkl
results/phase2/slab_results.pkl
results/phase3/adsorption_results.pkl
```

Required replay energy-ranking index:

```text
data/replay/gt_index_by_sid.pkl
data/replay/gt_index_by_system.pkl
```

Required checkpoint for current best all-pair baseline:

```text
runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay/args.json
runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay/ckpt_epochepoch=029.ckpt
```

Explicitly exclude from Dropbox:

```text
runs/**
logs/**
wandb/**
lightning_logs/**
**/*.log
**/*.out
**/*.err
**/last.ckpt
**/replay_viz/**
**/replay_buffer*.pkl
**/replay_metrics*.json
**/revalidated*.json
**/checkpoint_gpu*.pkl
**/*cache*.pkl
results/**/*.log
results/pristine_slabs/oc20dense.pkl
```

The only exceptions under `runs/` are the two all-pair ep29 files listed above.
The only exceptions under `results/` are the required pkl files listed in the
sections above.

Approximate sizes:

| Artifact | Size |
| --- | ---: |
| `data/processed` | 3.6G |
| `data/pkls/adsorbates.pkl` | 52K |
| `results/phase1/bulk_results.pkl` | 576M |
| `results/phase2/slab_results.pkl` | 11G |
| `results/phase3/adsorption_results.pkl` | 13G |
| `results/pristine_slabs/is2res.pkl` | 842M |
| `results/pristine_slabs/is2res.sid_index.pkl` | 17M |
| `results/pristine_slabs/oc20dense_uma.pkl` | 3.2M |
| `results/pristine_slabs/oc20dense.system_index.pkl` | 44K |
| `data/replay` | 72M |
| all-pair ep29 ckpt | 1.2G |

## Fast Dynamics Replay Usage

Before replay, ensure:

```bash
python - <<'PY'
import fast_dynamics
from fast_dynamics import UMAWrapper, prepare_batch_for_dynamics
print("fast_dynamics ok", fast_dynamics.__file__)
PY
```

Single-checkpoint replay path:

```bash
PYTHONPATH=AdsorbGen python scripts/replay_one_ckpt.py \
  --ckpt runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay/ckpt_epochepoch=029.ckpt \
  --gt-index data/replay/gt_index_by_sid.pkl \
  --train-lmdb data/processed/is2res_train.lmdb \
  --viz-root runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay/replay_viz \
  --buffer-path runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay/replay_buffer_ep30.pkl \
  --metrics-path runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay/replay_metrics_ep30.json \
  --epoch-tag 30 \
  --num-systems 500 \
  --num-placements 3 \
  --flow-steps 50 \
  --uma-max-steps 300 \
  --pristine-slabs results/pristine_slabs/is2res.pkl \
--pristine-index results/pristine_slabs/is2res.sid_index.pkl
```

4-GPU replay uses process-level sharding, one process per GPU:

```bash
GPUS="0 1 2 3" \
RUN_DIR=runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay \
CKPT=runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay/ckpt_epochepoch=029.ckpt \
NUM_SYSTEMS=500 \
NUM_PLACEMENTS=3 \
UMA_MAX_STEPS=300 \
bash scripts/run_replay_4gpu.sh
```

Each shard owns one GPU and runs batched FIRE through `fast_dynamics`.
`scripts/merge_replay_shards.py` merges shard buffers, metrics, and web-viz
outputs.

## Placement Preflight

`scripts/preflight_placement.py` is not required for normal training or
evaluation, but it is useful after moving servers. It samples processed LMDB
entries and checks whether fairchem `AdsorbateSlabConfig` can generate fresh
placements from `data/pkls/adsorbates.pkl`.

Run once after unpacking data:

```bash
PYTHONPATH=AdsorbGen python scripts/preflight_placement.py \
  --lmdb data/processed/is2res_train.lmdb \
  --ads-pkl data/pkls/adsorbates.pkl \
  --n 500
```

If this fails broadly, training with `prior_mode=random_heuristic` will also be
unreliable.

## Evaluation Reference Caveat

Training-time `sample_eval/dense/*` currently uses `oc20dense.lmdb` and usually
does not have a matching pristine slab DB. Its `surf_changed_rate` therefore
falls back to `pos_gt[tag != 2]` as the slab reference.

IS2RES post-hoc revalidation/replay can use:

```text
results/pristine_slabs/is2res.pkl
results/pristine_slabs/is2res.sid_index.pkl
```

When comparing Dense vs IS2RES, report the surface reference explicitly:

- `surface_ref=pos_gt_fallback`
- `surface_ref=pristine_is2res`
