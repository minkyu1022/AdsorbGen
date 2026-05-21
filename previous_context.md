# AdsorbGen Working Context

Last updated: 2026-05-21 12:48 KST

This note summarizes the current working state, decisions, generated data, and processes from the recent Codex/Claude-assisted work. It is meant to let the next session continue without re-discovering everything.

## Current User Intent

- Do not start new training jobs yet unless explicitly requested.
- Keep the code/data state understandable and avoid unnecessary legacy files.
- Prepare, but do not automatically run, these next experiments:
  - B-size model with CatFlow-style adsorbate center + relative-position head and adsorbate pair-distance loss.
  - Baseline1 adsorbate-only movable ablation: slab fixed at initial structure, only adsorbate atoms moved/lossed.
  - Both should use OC20 IS2RES train + val ID only, exclude OOD from training, and use MLIP-relaxed targets.

## Important Correction

I briefly launched the two 4-GPU training jobs before the user clarified not to start training yet. They were stopped immediately afterward.

Stopped runs:

- `/home/irteam/runs/B_center_rel_pairdist_mlip_id_unwrap_20260521`
- `/home/irteam/runs/B1_ads_only_center_rel_pairdist_mlip_id_unwrap_20260521`

These directories contain early `args.json`, `train.log`, `pid.txt`, and wandb scaffolding from the aborted launch. They are not meaningful trained checkpoints.

Current check after stopping: no active `adsorbgen.training.train_cli` processes.

## Repo State Highlights

Repository: `/home/irteam/AdsorbGen`

Important changes/additions made during this work:

- Started repo cleanup/refactor:
  - operational scripts are grouped under `scripts/{train,preprocess,replay,analysis}`
  - one-off OC20Dense/MLIP Pass utilities are under `experiments/`
  - legacy archive-only utilities were deleted instead of carried forward
  - core modules are partially split into `adsorbgen/{data,models,flow,evaluation,replay,training,inference}`
  - active code now imports the new subpackage paths directly, e.g. `adsorbgen.data.dataset`, `adsorbgen.models.dit`, `adsorbgen.replay.eval`, and `adsorbgen.training.train_cli`
  - old top-level wrappers such as `adsorbgen/dataset.py`, `adsorbgen/model.py`, `adsorbgen/eval_replay.py`, and `adsorbgen/train.py` were removed
  - stale tests that depended on removed displacement-flow APIs were rewritten for the current absolute-coordinate flow API

- Added `adsorbgen/data/pbc_unwrap.py`
  - Implements periodic adsorbate unwrapping and centering helpers.
  - API used by `adsorbgen/scripts/unwrap_preprocess.py`:
    - `PBC_XY`, `PBC_XYZ`
    - `load_adsorbate_reference_db`
    - `preprocess_entry_geometry`
    - `summarize_geometry_stats`

- `adsorbgen/scripts/unwrap_preprocess.py`
  - CLI to transform already-preprocessed LMDBs by unwrapping adsorbates and applying one rigid centering translation.
  - Verified with actual train/val LMDB samples.

- `adsorbgen/models/variants.py`
  - Added/verified B-size CatFlow center/relative-position variant:
    - `v0-ads-ref-catflow-center-rel`
    - About 101.94M parameters.
    - `use_ads_ref_pos=True`
    - `use_ads_specific_head=False`
    - `use_ads_center_rel_head=True`

- Added launcher:
  - `/home/irteam/baselines/launch_B_and_B1_center_rel_pairdist_id.sh`
  - It is prepared but should not be run until user asks.
  - Uses only:
    - `/home/irteam/data/processed_unwrap_centered_mlip_oc20_id/is2res_train.lmdb`
    - `/home/irteam/data/processed_unwrap_centered_mlip_oc20_id/is2res_val.lmdb`
  - Does not include OOD LMDBs.

Other scripts created earlier:

- `/home/irteam/AdsorbGen/experiments/2026-05-oc20dense/compute_oc20dense_mlip_relax.py`
- `/home/irteam/AdsorbGen/experiments/2026-05-oc20dense/merge_oc20dense_mlip_relax.py`
- `/home/irteam/AdsorbGen/experiments/2026-05-oc20dense/run_oc20dense_mlip_relax_4gpu.sh`
- `/home/irteam/AdsorbGen/experiments/2026-05-oc20dense/monitor_oc20dense_mlip_relax.py`
- `/home/irteam/AdsorbGen/experiments/2026-05-oc20dense/materialize_oc20dense_slab_refs.py`
- `/home/irteam/AdsorbGen/experiments/2026-05-mlip-pass/eval_mlip_pass_lbfgs_ood50.py`
- `/home/irteam/AdsorbGen/experiments/2026-05-mlip-pass/merge_mlip_pass_lbfgs_ood50.py`
- `/home/irteam/AdsorbGen/experiments/2026-05-mlip-pass/run_mlip_pass_lbfgs_ood50_8gpu.sh`

## Unwrap + Center Preprocessing

User provided the body of `pbc_unwrap.py`; I placed it at:

- `/home/irteam/AdsorbGen/adsorbgen/data/pbc_unwrap.py`

Small validation:

- 500 samples from `is2res_train.lmdb`: success, skipped 0.
- 500 samples from `is2res_val.lmdb`: success, skipped 0.
- Schema preserved: `pos`, `pos_relaxed`, `mlip_e_total`, `mlip_converged`, `sid`, `ads_id`, etc.

Full transformed ID-only MLIP target LMDBs were created at:

- `/home/irteam/data/processed_unwrap_centered_mlip_oc20_id/is2res_train.lmdb`
- `/home/irteam/data/processed_unwrap_centered_mlip_oc20_id/is2res_val.lmdb`

Reports:

- `/home/irteam/data/processed_unwrap_centered_mlip_oc20_id/is2res_train.report.json`
  - records/written: 460,328
  - skipped: 0
  - center mode: `relaxed_all`
  - pbc axes: `xy`
  - split_before: 1,414
  - split_after: 106
  - changed_records: 2,938
  - recentered_records: 460,328

- `/home/irteam/data/processed_unwrap_centered_mlip_oc20_id/is2res_val.report.json`
  - records/written: 24,943
  - skipped: 0
  - center mode: `relaxed_all`
  - pbc axes: `xy`
  - split_before: 83
  - split_after: 8
  - changed_records: 164
  - recentered_records: 24,943

Note: `split_after` uses a conservative diameter heuristic. Remaining nonzero count does not automatically mean the molecule is still wrong; it should be inspected if this matters.

## Current Process State

As of this note:

- No intended training should be running.
- No active `adsorbgen.training.train_cli` process should remain.
- GPUs were freed after stopping the accidental training launch.

If checking manually:

```bash
pgrep -af 'adsorbgen.training.train_cli' || true
nvidia-smi
```

## Training Plan Prepared, Not Running

Prepared launcher:

```bash
/home/irteam/baselines/launch_B_and_B1_center_rel_pairdist_id.sh
```

It would launch:

- GPUs 0-3:
  - Out: `/home/irteam/runs/B_center_rel_pairdist_mlip_id_unwrap_20260521`
  - Variant: `v0-ads-ref-catflow-center-rel`
  - `movable_mode=surface_ads`
  - `loss_surf_weight=1.0`
  - `loss_ads_weight=1.0`
  - `ads_pair_l1_weight=1.0`
  - batch size default: 96
  - epochs default: 100

- GPUs 4-7:
  - Out: `/home/irteam/runs/B1_ads_only_center_rel_pairdist_mlip_id_unwrap_20260521`
  - Same variant/loss/data.
  - `movable_mode=adsorbate_only`
  - `loss_surf_weight=0.0`
  - `loss_ads_weight=1.0`

Important: because training was aborted almost immediately, if this experiment is restarted later it may be cleaner to delete or rename the two aborted run directories first, or set new `MAIN_OUT` and `B1_OUT`.

## OC20-Dense MLIP Relaxation

Completed full OC20-dense MLIP relaxation:

- Summary:
  - `/home/irteam/data/replay/oc20dense_mlip_relax_summary.json`
- Full merged:
  - `/home/irteam/data/replay/oc20dense_mlip_relax.pkl`
- Global min by system:
  - `/home/irteam/data/replay/oc20dense_mlip_global_min_by_system.pkl`
- Shards:
  - `/home/irteam/data/replay/oc20dense_mlip_relax_shards`

Results:

- total records: 65,073 / 65,073
- converged: 62,621
- converged rate: about 96.23%

Settings used/confirmed in the surrounding discussion:

- UMA task: `oc20`
- UMA model family discussed: `uma-s-1p1`
- Earlier batched relaxation used FIRE for parallel work.
- Paper-comparable MLIP Pass later used ASE L-BFGS settings.

## Pristine Slab MLIP References

Existing slab refs were materialized into replay-friendly files:

- Source cache:
  - `/home/irteam/results/pristine_slabs/oc20dense_uma.pkl`
- Materialized:
  - `/home/irteam/data/replay/oc20dense_E_slab_only_by_slab.pkl`
  - `/home/irteam/data/replay/oc20dense_E_slab_only_by_system.pkl`
  - `/home/irteam/data/replay/oc20dense_E_slab_only_summary.json`

Coverage:

- 967 / 967 unique slabs
- 973 dense systems
- 1 unconverged slab in source cache

Meaning of "materialized" in prior discussion:

- The source pristine slab cache already existed in a less directly usable form.
- I converted/exposed it into lookup files keyed by slab/system so downstream adsorption-energy code can load it without re-relaxing slabs.

## OC20-Dense Split Membership

Computed split membership:

- `/home/irteam/data/replay/oc20dense_oc20_split_membership.json`

OC20-dense:

- systems: 973
- samples: 65,073

Exact pair ID:

- systems: 244
- samples: 15,450

OOD groups:

- val OOD ads: 244 systems, 15,751 samples
- val OOD cat: 238 systems, 16,233 samples
- val OOD both: 247 systems, 17,639 samples

Not train+val ID exact pair:

- systems: 729
- samples: 49,623

For the extracted 100-system global-min cover:

- 26 ID
- 25 OOD ads
- 26 OOD cat
- 23 OOD both

## AdsorbSample / MLIP Pass Clarifications

Paper file inspected:

- `/home/irteam/AdsorbSample/323_Diffusion_Sampling_of_Adso.pdf`

Key interpretation:

- `E_MLIP_reference_min(s)` / `E*_{theta,s}` is the MLIP reference minimum for that system from the reference candidate set, not the minimum among generated model samples.
- MLIP Pass is computed after applying the paper's valid-candidate filter.
- Formula uses valid candidates only:
  - `c_s = |{X in S_hat_s^val : E_theta(X; c) - E*_{theta,s} <= eps_succ}|`
  - `MLIP Pass@k(s) = 1 - C(n_s - c_s, k) / C(n_s, k)`
  - with edge cases `c_s=0 -> 0`, `n_s-c_s < k -> 1`.

Paper relaxation settings confirmed from Appendix B.2:

- UMA-S-1.1
- OC20 task head
- ASE L-BFGS
- `fmax = 0.01 eV/Angstrom`
- max steps: 300
- maxstep/displacement cap: 0.04 Angstrom
- history size/memory: 50
- damping: 1.0
- alpha: 70 eV/Angstrom^2

## OOD-50 MLIP Pass Sanity Check

Completed:

- Output:
  - `/home/irteam/data/replay/mlip_pass_lbfgs_ood50/summary.json`

Run window:

- 2026-05-21 07:18:01 to 09:13:24 KST

Design:

- Choose 50 systems from the 74 OOD systems in the 100-system global-min cover.
- Generate/evaluate 100 candidates per system.
- Total candidates: 5,000.
- Relax/evaluate with paper-style L-BFGS settings.

Results:

- systems: 50
- candidates: 5,000
- converged_rate: 0.8824
- valid_rate: 0.7198
- success_sample_rate: 0.2436
- MLIP Pass@1: 0.2436
- MLIP Pass@2: 0.3022666667
- MLIP Pass@5: 0.3569361360
- MLIP Pass@10: 0.3906914158

## Replay / Self-Improvement Decisions

Important replay-success criterion change:

- Earlier strict threshold used `GT E_ads - 0.05 eV`.
- User asked to remove that margin because all references are MLIP-based.
- Later discussion preferred per-unique-system mean adsorption energy from data as the threshold:
  - add generated candidate to buffer if its adsorption energy is lower than the mean for that unique system.

Replay status notes from earlier:

- H200 pair-dist-loss checkpoint epoch 99 exists and was used for replay comparisons.
- `collect_predictions` is not necessary for the user's requested success-rate table once inline counters exist.
- Keeping prediction PKLs is only useful for ad-hoc later analysis such as `E_pred - E_gt`, per-placement counts, alternate margins, or failure patterns.

## Baselines Discussed

Baseline 1:

- Adsorbate-only movable ablation.
- Slab stays fixed at initial structure.
- Adsorbate atoms move.
- Should use same CatFlow center/rel + ads pair-dist loss in the pending new training.

Baseline 2:

- Random-heuristic placement only.
- No model inference before UMA relaxation.
- Purpose: compare self-improvement success rate among converged samples against model-generated structures.

Baseline 3:

- AdsorbDiff baseline.
- AdsorbDiff repo was cloned.
- Checkpoint downloaded:
  - `/home/irteam/baselines/baseline3_adsorbdiff/checkpoints/PT_zeroshot_painn.pt`
- Still needs careful integration if continued.

Baseline workspace:

- `/home/irteam/baselines`

## Data Upload / Dropbox Context

Dropbox remote was configured via `rclone`.

Important distinction discussed:

- `processed_original_lmdbs`: original/Dataset-style LMDB targets.
- `processed_mlip_oc20`: MLIP-relaxed target LMDBs, aligned to UMA OC20 reference scale.
- User later renamed/uses `/home/irteam/data/processed` as the MLIP target data location.

The current training-prep output is separate:

- `/home/irteam/data/processed_unwrap_centered_mlip_oc20_id`

This folder contains only ID train/val transformed for unwrap+center.

## Model Size Discussion

Previous H200 ads pair-dist-loss model:

- Run:
  - `/home/irteam/runs/H200_ads_pair_dist_loss`
- Variant:
  - `v0-ads-ref-adshead-2x`
- Size:
  - about 205.93M parameters
- Main dimensions:
  - atom/token single dim: 640
  - pair dim: 320
  - depths: 2 / 22 / 2
  - heads: 5 / 10 / 5

B-size option:

- Variant family:
  - `v0-ads-ref-adshead`
  - and now center/rel version `v0-ads-ref-catflow-center-rel`
- Size:
  - about 101.94M parameters
- Main dimensions:
  - atom/token single dim: 512
  - pair dim: 256
  - depths: 2 / 16 / 2
  - heads: 4 / 8 / 4

Reason:

- Smaller model while preserving approximate architectural proportions.
- Intended to allow larger batch size than the 2x model.

## Files To Be Careful With

Do not delete without explicit confirmation:

- `/home/irteam/data/replay/oc20dense_mlip_relax.pkl`
- `/home/irteam/data/replay/oc20dense_mlip_global_min_by_system.pkl`
- `/home/irteam/data/replay/oc20dense_E_slab_only_by_slab.pkl`
- `/home/irteam/data/replay/oc20dense_E_slab_only_by_system.pkl`
- `/home/irteam/data/replay/mlip_pass_lbfgs_ood50/summary.json`
- `/home/irteam/data/processed_unwrap_centered_mlip_oc20_id/*.lmdb`
- `/home/irteam/AdsorbGen/adsorbgen/data/pbc_unwrap.py`
- `/home/irteam/AdsorbGen/adsorbgen/scripts/unwrap_preprocess.py`

## Suggested Next Step

Before restarting any training:

1. Decide whether to keep/delete the aborted run dirs from the accidental launch.
2. Confirm whether `center_mode=relaxed_all` is acceptable for the final training data.
3. Confirm whether the validation LMDB should be `is2res_val.lmdb` or a separate OC20-dense validation file.
4. Then run, only on explicit user request:

```bash
/home/irteam/baselines/launch_B_and_B1_center_rel_pairdist_id.sh
```
