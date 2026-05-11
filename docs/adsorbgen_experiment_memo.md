# AdsorbGen Experiment Memo

## 2026-05-11 Current Full-Run Status

All GPUs are idle; no `adsorbgen.train` processes are currently running.

## all-pair auxiliary distance loss

Run: `full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay`

Configuration:

- `arch=v1`
- `variant=v0-ads-ref`
- `prediction_type=x1`
- `loss_type=l1`
- `loss_surf_weight=1.0`
- `loss_ads_weight=1.0`
- `ads_pair_l1_weight=1.0`
- `ads_bond_l1_weight=0.0`
- `ads_nonbonded_clash_weight=0.0`
- `sample_eval_max_samples=1000`
- `sample_eval_steps=20`
- no replay

Completion:

- Completed through epoch 29.
- Checkpoints exist for epochs 0-29 plus `last.ckpt`.

Dense sample-eval summary from local W&B history:

| epoch window | valid | overlap | dissoc | surf_changed | desorbed | any_anomaly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-29 mean | 0.6024 | 0.0716 | 0.2722 | 0.1042 | 0.0026 | 0.3976 |
| 20-29 mean | 0.6747 | 0.0538 | 0.1999 | 0.0784 | 0.0019 | 0.3253 |
| 25-29 mean | 0.6834 | 0.0502 | 0.1806 | 0.0788 | 0.0008 | 0.3166 |
| epoch 29 | 0.7720 | 0.0200 | 0.1200 | 0.0530 | 0.0000 | 0.2280 |

Interpretation:

- The all-pair auxiliary is the strongest intervention so far on dense
  sample-eval validity.
- It sharply reduces overlap to low single digits by the final epoch.
- Dissociation remains the largest residual invalid contributor, but is much
  lower than earlier no-auxiliary runs.
- Epoch 29 is also the best epoch by strict valid rate, but the 25-29 mean is
  the fairer stability estimate.

## Dense sample-eval comparison table

Protocol for this table:

- Dense validation sample-eval from local W&B history.
- `sample_eval_max_samples=1000`
- `sample_eval_steps=20`
- no final refine
- one generated sample per system
- Main table uses epoch 0-20 mean when available.

| run | 0-20 mean valid | overlap | dissoc | surf_changed | desorbed | intercalated | any_anomaly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control L1 | 0.241 | 0.363 | 0.693 | 0.081 | 0.006 | 0.050 | 0.759 |
| dynpair L1 | 0.225 | 0.358 | 0.714 | 0.076 | 0.008 | 0.050 | 0.775 |
| control L2 | 0.147 | 0.392 | 0.702 | 0.408 | 0.015 | 0.077 | 0.853 |
| dynpair L2 | 0.144 | 0.401 | 0.700 | 0.416 | 0.015 | 0.082 | 0.856 |
| lDDT 0.5 | 0.224 | 0.287 | 0.718 | 0.086 | 0.003 | 0.042 | 0.776 |
| lDDT 1.0 | 0.233 | 0.307 | 0.696 | 0.092 | 0.003 | 0.040 | 0.767 |
| bond+clash | 0.340 | 0.261 | 0.579 | 0.075 | 0.004 | 0.027 | 0.660 |
| all-pair | 0.569 | 0.080 | 0.307 | 0.115 | 0.003 | 0.083 | 0.431 |

Final-epoch reference:

| run | final epoch | final valid | overlap | dissoc | surf_changed | desorbed | intercalated | any_anomaly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control L1 | 45 | 0.170 | 0.592 | 0.681 | 0.036 | 0.003 | 0.022 | 0.830 |
| dynpair L1 | 45 | 0.158 | 0.802 | 0.689 | 0.061 | 0.017 | 0.003 | 0.842 |
| control L2 | 21 | 0.206 | 0.335 | 0.676 | 0.254 | 0.004 | 0.036 | 0.794 |
| dynpair L2 | 21 | 0.119 | 0.376 | 0.575 | 0.581 | 0.007 | 0.030 | 0.881 |
| lDDT 0.5 | 29 | 0.188 | 0.418 | 0.685 | 0.057 | 0.001 | 0.251 | 0.812 |
| lDDT 1.0 | 29 | 0.195 | 0.187 | 0.728 | 0.056 | 0.001 | 0.151 | 0.805 |
| bond+clash | 18 | 0.381 | 0.188 | 0.569 | 0.040 | 0.003 | 0.024 | 0.619 |
| all-pair | 29 | 0.772 | 0.020 | 0.120 | 0.053 | 0.000 | 0.055 | 0.228 |

Interpretation:

- All-pair is the only run that materially changes the failure profile rather
  than only moving small percentages around.
- Overlap falls from roughly 30-40% in prior baselines to 8.0% on the 0-20
  mean and 2.0% at epoch 29.
- Dissociation also drops from roughly 70% in the control/dynpair/lDDT runs to
  30.7% on the 0-20 mean and 12.0% at epoch 29.
- The remaining caveat is variance: epoch 29 is the best point, so final
  77.2% should be treated as a promising checkpoint result, while the 0-20
  and 20-29 means are better estimates of average behavior.

## IS2RES revalidation comparison table

Protocol for the main table:

- Source directory: `runs/is2res_k1_ep0_20_stride5_20steps_20260509`
- IS2RES validation subset, 1000 generated samples per checkpoint
- `sample_eval_steps=20`
- no final refine
- one generated sample per system
- Epochs available: 0, 5, 10, 15, 20
- all-pair row was backfilled on 2026-05-11 with the same protocol.
- all-pair epoch 29 was additionally evaluated as a single-checkpoint
  final-epoch check because the all-pair run completed through epoch 29.
- Baseline final checkpoints were backfilled on 2026-05-11 under the same
  20-step/no-refine IS2RES protocol in
  `runs/is2res_final_20steps_20260511`.

| run | 0-20 stride5 mean valid | overlap | dissoc | surf_changed | desorbed | intercalated | any_anomaly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control L1 | 0.212 | 0.414 | 0.724 | 0.097 | 0.004 | 0.034 | 0.788 |
| dynpair L1 | 0.187 | 0.441 | 0.736 | 0.098 | 0.005 | 0.071 | 0.813 |
| lDDT 0.5 | 0.146 | 0.358 | 0.767 | 0.184 | 0.004 | 0.097 | 0.854 |
| lDDT 1.0 | 0.180 | 0.396 | 0.741 | 0.104 | 0.004 | 0.082 | 0.820 |
| control L2 | 0.090 | 0.556 | 0.759 | 0.489 | 0.010 | 0.056 | 0.910 |
| dynpair L2 | 0.083 | 0.491 | 0.760 | 0.503 | 0.005 | 0.063 | 0.917 |
| all-pair | 0.404 | 0.106 | 0.471 | 0.160 | 0.003 | 0.090 | 0.596 |

Epoch 20 reference from the original stride-5 sweep:

| run | epoch | valid | overlap | dissoc | surf_changed | desorbed | intercalated | any_anomaly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control L1 | 20 | 0.129 | 0.731 | 0.734 | 0.143 | 0.003 | 0.112 | 0.871 |
| dynpair L1 | 20 | 0.146 | 0.784 | 0.743 | 0.149 | 0.004 | 0.035 | 0.854 |
| lDDT 0.5 | 20 | 0.087 | 0.658 | 0.743 | 0.569 | 0.009 | 0.023 | 0.913 |
| lDDT 1.0 | 20 | 0.179 | 0.570 | 0.731 | 0.159 | 0.006 | 0.050 | 0.821 |
| control L2 | 20 | 0.008 | 0.799 | 0.741 | 0.955 | 0.003 | 0.067 | 0.992 |
| dynpair L2 | 20 | 0.015 | 0.558 | 0.737 | 0.944 | 0.010 | 0.080 | 0.985 |
| all-pair | 20 | 0.457 | 0.098 | 0.477 | 0.102 | 0.003 | 0.022 | 0.543 |

True final checkpoints under the same IS2RES protocol:

| run | final epoch | valid | overlap | dissoc | surf_changed | desorbed | intercalated | any_anomaly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control L1 | 45 | 0.184 | 0.592 | 0.736 | 0.061 | 0.009 | 0.027 | 0.816 |
| dynpair L1 | 45 | 0.155 | 0.750 | 0.746 | 0.080 | 0.021 | 0.019 | 0.845 |
| lDDT 0.5 | 29 | 0.136 | 0.536 | 0.732 | 0.078 | 0.002 | 0.179 | 0.864 |
| lDDT 1.0 | 29 | 0.192 | 0.217 | 0.710 | 0.080 | 0.002 | 0.143 | 0.808 |
| control L2 | 21 | 0.167 | 0.427 | 0.723 | 0.284 | 0.016 | 0.016 | 0.833 |
| dynpair L2 | 21 | 0.073 | 0.615 | 0.717 | 0.590 | 0.006 | 0.025 | 0.927 |
| all-pair | 29 | 0.683 | 0.023 | 0.238 | 0.061 | 0.002 | 0.040 | 0.317 |

all-pair by epoch under the same IS2RES protocol:

| epoch | valid | overlap | dissoc | surf_changed | desorbed | intercalated | any_anomaly |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.206 | 0.131 | 0.607 | 0.145 | 0.001 | 0.314 | 0.794 |
| 5 | 0.228 | 0.177 | 0.575 | 0.422 | 0.004 | 0.010 | 0.772 |
| 10 | 0.558 | 0.052 | 0.332 | 0.072 | 0.004 | 0.076 | 0.442 |
| 15 | 0.571 | 0.073 | 0.362 | 0.059 | 0.003 | 0.028 | 0.429 |
| 20 | 0.457 | 0.098 | 0.477 | 0.102 | 0.003 | 0.022 | 0.543 |
| 29 | 0.683 | 0.023 | 0.238 | 0.061 | 0.002 | 0.040 | 0.317 |

Separate 50-step + final-refine IS2RES files exist only for control/dynpair
raw checkpoints, so they should not be mixed with the all-run table above.

## branch-wise typed pair DiT

Run: `full_v0_ads_ref_x1_l1_allpairL1ref1_branchpair_noreplay`

Purpose:

- Keep the successful DiT + all-pair auxiliary setup.
- Add branch-wise typed pair conditioning inside the existing DiT pair bias.
- This is not the pure GNN denoiser; coordinate prediction and DiT blocks are
  unchanged.

Variant:

- `arch=v1`
- `variant=v0-ads-ref-branchpair`
- `use_typed_pair_features=True`
- `typed_pair_include_bulk=True`

Branch conditioning implemented in `AdsorbGen/adsorbgen/model.py`:

- `ads_ads`: all adsorbate internal directed pairs, no cutoff.
- `ads_bond`: covalent adsorbate graph inferred from `ads_ref_pos`, no cutoff.
- `ads_surface`: adsorbate-surface contacts within 6 A.
- `surface_surface`: surface-surface local graph within 6 A.
- `surface_bulk`: surface-bulk read-only context within 6 A.

Implementation notes:

- Branch features are fused with a learned per-pair gate and added as a
  residual to the existing pair bias.
- Branch projections/topology/bond embeddings are zero-initialized, so the
  model starts from the previous DiT pair-bias behavior and learns the new
  branch residual gradually.

Launch:

- Launched 2026-05-11 UTC in tmux session `aux_allpair_branchpair`.
- `GPUS=0,1,2,3 DEVICES=4`
- `BATCH=16`
- `EPOCHS=30`
- `USE_REPLAY=0`
- `ADS_PAIR_L1_WEIGHT=1.0`
- `ADS_BOND_L1_WEIGHT=0.0`
- `ADS_NONBONDED_CLASH_WEIGHT=0.0`
- `SAMPLE_EVAL_MAX=1000`
- `SAMPLE_EVAL_STEPS=20`

Startup verification:

- `py_compile` passed for modified files.
- Synthetic forward passed with finite output and non-movable clamp.
- Full run DDP initialized, sanity check passed, and epoch 0 training started.
- Stopped on 2026-05-11 UTC during epoch 2 to free GPU0-3 for the more direct
  all-pair + ads-specific-output-head ablation. This run had not reached a
  decisive sample-eval regime.

## all-pair + ads-specific output head DiT

Run: `full_v0_ads_ref_x1_l1_allpairL1ref1_adshead_noreplay`

Purpose:

- Directly test whether splitting the final x1 coordinate projection for
  adsorbate atoms improves the already strong DiT + all-pair auxiliary setup.
- This isolates output-head negative transfer between adsorbate geometry
  prediction and movable-surface relaxation. It does not add branch-wise pair
  conditioning.

Variant:

- `arch=v1`
- `variant=v0-ads-ref-adshead`
- `use_ads_ref_pos=True`
- `use_ads_specific_head=True`
- `use_typed_pair_features=False`

Loss and protocol:

- `ADS_PAIR_L1_WEIGHT=1.0`
- `ADS_BOND_L1_WEIGHT=0.0`
- `ADS_NONBONDED_CLASH_WEIGHT=0.0`
- `LOSS_TYPE=l1`
- `SAMPLE_EVAL_MAX=1000`
- `SAMPLE_EVAL_STEPS=20`
- `USE_REPLAY=0`

Launch:

- Launched 2026-05-11 UTC in tmux session `aux_allpair_adshead`.
- `GPUS=0,1,2,3 DEVICES=4`
- `BATCH=16`
- `EPOCHS=30`

Startup verification:

- DDP initialized on 4 ranks.
- Model summary shows both `model.out_proj` and `model.ads_out_proj`.
- Sanity check passed and epoch 0 training started.

## all-pair replay probe, ep29 checkpoint

Run source: `full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay`

Checkpoint:

- `runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay/ckpt_epochepoch=029.ckpt`

Purpose:

- Probe whether the strong all-pair checkpoint can produce any replayable
  candidates after 50-step flow inference plus UMA relaxation.
- Use web-UI-compatible replay visualization artifacts.

Fast dynamics sanity:

- `fast_dynamics` bridge tests passed: `9 passed`.
- Real UMA wrapper/dynamics tests on GPU5 passed: `7 passed`.
- On real IS2RES adsorbate/slab samples, `fast_dynamics.UMAWrapper` matched
  `FAIRChemCalculator` energy/force output to about `1e-5`.
- `FixedAtomsHook` kept fixed atoms at zero displacement in a short UMA FIRE
  smoke test.

Launch history:

- A temporary single-GPU probe on GPU4 was stopped before completion to free
  GPU4 for sharded replay.
- 4-GPU sharded replay launched 2026-05-11 UTC in tmux session
  `replay4gpu_ep30`.
- Command surface: `scripts/run_replay_4gpu.sh`.
- GPUs: `4 5 6 7`.
- `EPOCH_TAG=30`, `NUM_SYSTEMS=500`, `NUM_PLACEMENTS=3`.
- `replay_one_ckpt.py` default `--flow-steps=50` is used.
- Each shard evaluates `125 / 500` sampled systems and starts from flow
  sampling before UMA relaxation.

Outputs:

- Main log:
  `runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay/replay_4gpu_ep30.log`
- Per-shard logs:
  `runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay/replay_shards/shard_*/worker.log`
- Final merged viz target:
  `runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay/replay_viz/ep30`
- Final merged metrics target:
  `runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay/replay_metrics_ep30.json`
