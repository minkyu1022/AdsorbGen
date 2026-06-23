# AdsorbGen Working Context

Last updated: 2026-06-23 10:30 KST

## Latest Critical Handoff Context, 2026-06-23

This section supersedes older historical notes below when there is any conflict.

### User Requests That Must Not Be Forgotten

- GPU idle time is unacceptable. For long multi-stage jobs, do not trust a single
  watcher. Always verify actual GPU utilization with `nvidia-smi`, and attach a
  guard/watchdog that restarts or alerts if the expected process disappears
  before completion.
- For any generation -> relaxation transition, explicitly confirm the next stage
  has started by checking process list, GPU utilization, and logs. Do not only
  check that a launcher returned.
- Headline pass@k must use `mlip_pass@k`; `valid_mlip_pass@k` is diagnostic only.
- Surface-change validity must use the bare relaxed slab reference. Do not fall
  back to `pos_gt`; if the bare slab reference is missing, fail loudly.
- When handing work to another server/agent, do not only patch local code. Write a
  clear prompt/guideline file that the other agent can read and act on.

### Current Running Job: 30k Self-Improvement Loop0 Full Replay

Output directory:

- `/home1/irteam/data/replay/id30k_self_loop0_epoch999_full_replay_sdeheun200_uniform_x5_20260623_034414`

Generation status:

- Flow generation completed successfully.
- `flow_summary.json` exists.
- `flow_jobs`: 118 shards.
- Total candidates: 30,000 systems x 5 placements = 150,000 candidates.
- Inference settings:
  - checkpoint:
    `/home1/irteam/runs/training/id31k_strictclean_x1_SI_vloss_eta_102M_sigma0p1_w0p5_uma1p2_uniform_ep2000/ckpt_epochepoch=999.ckpt`
  - SI + SDE + Heun, 200 steps
  - uniform time schedule
  - `si_gamma_sigma=0.1`
  - `si_epsilon_scale=0.01`
  - prior: random heuristic

Relaxation status as of 2026-06-23 10:28 KST:

- Relaxation is running.
- Main relax PID: `3204342`
- Guard PID: `3218114`
- ETA/monitor PID: `3035346`
- `relax_summary.json`: not yet present.
- `relax_results`: 8 shard JSONs completed so far.
- GPU0-7 are in use; instantaneous utilization varies by shard boundary but
  should not remain idle.
- Logs:
  - `/home1/irteam/data/replay/id30k_self_loop0_epoch999_full_replay_sdeheun200_uniform_x5_20260623_034414/logs/relax.direct.setsid.log`
  - `/home1/irteam/data/replay/id30k_self_loop0_epoch999_full_replay_sdeheun200_uniform_x5_20260623_034414/logs/relax_guard.log`
  - `/home1/irteam/data/replay/id30k_self_loop0_epoch999_full_replay_sdeheun200_uniform_x5_20260623_034414/logs/live_eta_monitor.log`

Important incident:

- Generation finished around 2026-06-23 09:06 KST, but the original watcher died
  after only logging that it was watching generation. GPU went idle instead of
  entering relaxation.
- Relaxation had to be relaunched manually with `setsid`.
- Permanent launch script:
  - `/home1/irteam/launch_id30k_loop0_relax_direct.sh`
- Guard script:
  - `/home1/irteam/guard_id30k_loop0_relax.sh`
- The guard checks for missing relax process before `relax_summary.json` exists,
  restarts it via `setsid`, and logs repeated status.

Commands for a new session to check first:

```bash
OUT=/home1/irteam/data/replay/id30k_self_loop0_epoch999_full_replay_sdeheun200_uniform_x5_20260623_034414
nvidia-smi
ps -eo pid,ppid,etime,stat,cmd | rg 'two_stage_full_replay.py relax|guard_id30k_loop0_relax|monitor_id30k_loop0_eta'
test -f "$OUT/relax_summary.json" && echo RELAX_DONE || echo RELAX_RUNNING
find "$OUT/relax_results" -maxdepth 1 -name 'relax_*.json' | wc -l
tail -n 40 "$OUT/logs/relax_guard.log"
tail -n 40 "$OUT/logs/relax.direct.setsid.log"
```

### What To Do When Relaxation Completes

When `relax_summary.json` appears:

1. Report completion and key throughput/convergence stats.
2. Build/update the self-improvement buffer for the same 30k train subset.
3. Buffer acceptance criterion currently intended:
   - geometry-valid candidate
   - finite post-relaxation energy
   - converged under UMA relaxation
   - post-relax energy improves system's current best with 0.05 eV tolerance,
     per the user's latest request to use `0.05eV` instead of `0.1eV`.
4. Preserve the original loop0 reference energy per system, so loop metrics can
   track whether repeated loops move post-gap below the starting reference.
5. Buffer entries must go through the existing unwrap/centering preprocessing
   path before training.
6. Prepare loop metrics for plotting after several loops:
   - new best count
   - post-gap vs original reference energy
   - valid rate
   - convergence rate
   - throughput
   - OOD50 pass@k once per loop
7. Next train loop was discussed as possibly 500 epochs after loop0, but if
   uncertain, report status and ask briefly before launching.

### Forward Execution Plan After Current Relaxation

The next session should continue in this order.

#### 1. Finish and verify the current relaxation

- Keep monitoring until:
  - `relax_summary.json` exists, and
  - all expected relaxation shards are written under `relax_results/`.
- Do not assume completion from process exit alone. Check both files and logs.
- If the relax process exits before `relax_summary.json`, inspect
  `relax.direct.setsid.log`, keep GPUs occupied if possible, and restart with:

```bash
setsid -f /home1/irteam/launch_id30k_loop0_relax_direct.sh \
  >> /home1/irteam/data/replay/id30k_self_loop0_epoch999_full_replay_sdeheun200_uniform_x5_20260623_034414/logs/relax.direct.setsid.log 2>&1
```

- Ensure the guard remains running:

```bash
setsid -f /home1/irteam/guard_id30k_loop0_relax.sh >/dev/null 2>&1
```

#### 2. Summarize current replay results

After completion, report at minimum:

- total candidates
- converged rate
- geometry-valid rate if available
- relaxation throughput
- average/median/best relaxation steps
- average/best post-relax energy gap if available
- how many candidates/systems are eligible for buffer insertion

The current full replay output format is from
`/home1/irteam/AdsorbGen/geoopt/two_stage_full_replay.py`.
It writes `relax_results/relax_*.json` and saved result pkl files when
`--save-result-pkl` is set.

Important: older self-improvement scripts may expect old replay-worker files
such as `rows_shard*.pkl`, `candidate_shard*.pkl`, or `success_shard*.pkl`.
Do not run them blindly. First inspect the actual `relax_results` schema and, if
needed, write a small adapter that converts current two-stage replay output into
the schema expected by the metric/materialization scripts.

Relevant scripts to inspect/reuse:

- `/home1/irteam/AdsorbGen/scripts/replay/record_self_improve_loop_metrics.py`
- `/home1/irteam/AdsorbGen/scripts/replay/materialize_window_lmdb.py`
- `/home1/irteam/AdsorbGen/scripts/replay/plot_self_improve_loop_metrics.py`
- `/home1/irteam/AdsorbGen/scripts/replay/wait_then_materialize_window_and_resume.sh`

Caution: `wait_then_materialize_window_and_resume.sh` still contains older
`/home/irteam` default paths and should not be launched as-is on this server.
Use it as a reference, or patch all paths and expected replay-file names first.

#### 3. Record loop metrics

Use or adapt `record_self_improve_loop_metrics.py` to write:

- metrics root:
  `/home1/irteam/data/replay/id30k_self_improve_loop_metrics_20260623`
- `loop_idx=0`
- replay dir:
  `/home1/irteam/data/replay/id30k_self_loop0_epoch999_full_replay_sdeheun200_uniform_x5_20260623_034414`
- train run:
  `/home1/irteam/runs/training/id31k_strictclean_x1_SI_vloss_eta_102M_sigma0p1_w0p5_uma1p2_uniform_ep2000`
- train ckpt:
  `/home1/irteam/runs/training/id31k_strictclean_x1_SI_vloss_eta_102M_sigma0p1_w0p5_uma1p2_uniform_ep2000/ckpt_epochepoch=999.ckpt`
- window/tolerance:
  `0.05 eV`

If `record_self_improve_loop_metrics.py` cannot read current replay output
directly, patch or wrap it rather than forcing old file names.

#### 4. Materialize next-loop training LMDB

Build a moving-window/self-improvement LMDB using:

- original strict-clean 30k train LMDB:
  `/home1/irteam/data/uma_s_1p2_references/processed/id_strict_clean_subsets_seed20260622/id_strict_clean_train30000_seed20260622.lmdb`
- replay candidates from the current relaxation result
- window/tolerance:
  `0.05 eV`

Every new LMDB must be unwrapped and centered before training. Do not train on a
raw materialized LMDB that skipped unwrap/centering.

Expected safe pattern:

1. Materialize raw next-loop LMDB.
2. Run `adsorbgen.scripts.unwrap_preprocess` on it.
3. Verify report JSON:
   - row count
   - skipped count
   - anomaly mask presence
   - sample schema contains `pos`, `pos_relaxed`, `mlip_e_total`/`y_relaxed`,
     `sid`, `ads_id`, and self-improvement metadata.

#### 5. Launch next training loop only after data verification

The current best training family is:

- x1 SI
- v loss
- denoiser enabled
- `gamma_sigma=0.1`
- `si_denoiser_loss_weight=0.5`
- `train-time-sampling=uniform`
- initial slab x0, not bare slab x0
- ads pair distance loss enabled
- UMA-s-1p2 evaluation/reference scale
- 8 GPUs if available

Base launcher to use as reference:

- `/home1/irteam/AdsorbGen/scripts/train/launch_id_strict_clean_x1_si_uniform_8gpu_ep500.sh`

Do not blindly use its default `TRAIN_LMDB`; override it to the newly
materialized unwrap/centered self-improvement LMDB. Preserve:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `--pristine-slabs` and `--pristine-index` pointing to bare slab references
- validation with bare slab refs, no fallback to `pos_gt`
- wandb resume/name consistency if resuming an existing intended run

Discussed epoch plan:

- loop0 model was trained to epoch 999 before replay.
- For loop1 and later, 500 epochs may be enough, but this is not final. If the
  user is not present and GPUs would otherwise idle, prefer launching the agreed
  500-epoch next loop after data verification and report immediately.

#### 6. OOD50 pass@k once per loop

After each training loop, run OOD50 pass@k with the current paper-style headline
metric:

- `mlip_pass@k` over all 100 candidates per system
- invalid candidates count as failures
- `valid_mlip_pass@k` diagnostic only

Use default LBFGS for the main loop-comparison table unless the user explicitly
asks for strict:

- default: `fmax=0.05`, `maxstep=0.2`, `memory=100`, `max_steps=300`
- UMA model/task: `uma-s-1p2`, `oc20`

#### 7. Plot trends after several loops

After about three loops, generate plots using:

- `/home1/irteam/AdsorbGen/scripts/replay/plot_self_improve_loop_metrics.py`

Expected plots:

- energy gap to initial reference vs loop
- accepted buffer/new-best rates vs loop
- validity/convergence/steps/throughput vs loop
- OOD50 `mlip_pass@k` vs loop

### Current Pass@k Rule

For current OOD50/30k-loop reports, the headline pass@k is `mlip_pass@k`.

- Each system has `n_s = 100` generated candidates.
- Invalid candidates are failures. They are not removed from the denominator.
- A candidate is successful only if it is geometrically valid and its
  post-relaxation energy gap is `<= 0.1 eV`.
- For each system:
  - `c_s = # successful candidates among all 100 generated candidates`
  - `MLIP Pass@k(s) = 1 - C(100 - c_s, k) / C(100, k)`
  - report the average over systems.
- `valid_mlip_pass@k` is valid-only diagnostic information. Do not use it as the
  headline metric unless explicitly asked.
- This is not candidate-level success fraction except at `k=1`, and it is not
  "rank valid candidates by energy and check top-k".

Recent corrected loop0 epoch999 OOD50 result under this rule:

- Output:
  `/home1/irteam/data/replay/self_loop0_epoch999_ood50_passk_uma1p2_defaultlbfgs_sdeheun200_uniform_20260623_0230/summary.json`
- all-candidate `mlip_pass@1/2/5/10`: 17.28 / 27.11 / 42.53 / 53.14%
- valid-only diagnostic `valid_mlip_pass@1/2/5/10`: 21.81 / 32.67 / 47.83 / 57.07%
- valid rate: 70.36%
- converged rate: 98.80%

### FK Steering Agent Handoff

The FK steering package originally encouraged the other agent to read
`valid_mlip_pass@k` as the leaderboard metric. That is wrong for the current
project rule.

Prompt/guideline files to give another server/agent:

- `/home1/irteam/FK_PASSK_CORRECTION_PROMPT_20260623.md`
- `/home1/irteam/FK_PASSK_GUIDELINE_20260623.md`

Instruction to the other agent:

- Stop using `valid_mlip_pass@k` as headline.
- Recompute/re-merge existing FK results from raw shard outputs if possible,
  without rerunning GPU jobs unless raw rows are missing.
- Rank primarily by `mlip_pass@10`, then `mlip_pass@5`.
- Keep `valid_mlip_pass@k`, valid rate, valid success rate, pre-gap, post-gap,
  and relaxation steps as diagnostic columns.

### Notes About Older Content Below

Much of the older content below uses `/home/irteam/...` paths and describes
earlier 2025/May experiments. The current active workspace is `/home1/irteam`.
Do not assume old "Current Process State" or "Suggested Next Step" sections are
current unless they match the latest section above.

This note summarizes the current working state, decisions, generated data, and processes from the recent Codex/Claude-assisted work. It is meant to let the next session continue without re-discovering everything.

## Critical Pass@k Reporting Rule

- For all current AdsorbGen OOD50/30k-loop reports, the primary MLIP pass@k metric is computed over all `n_s = 100` generated candidates per system.
- Invalid candidates are failures. They are not removed from the denominator.
- For each system:
  - `c_s = number of geometrically valid candidates with post-relaxation energy gap <= 0.1 eV`.
  - `MLIP Pass@k(s) = 1 - C(100 - c_s, k) / C(100, k)`.
  - Report the average of `MLIP Pass@k(s)` over systems.
- In diagnostic JSON files, top-level `mlip_pass@k` means this primary all-candidate metric.
- Top-level `valid_mlip_pass@k` is valid-only and is only a secondary diagnostic. Do not use it as the headline pass@k unless the user explicitly asks for valid-only pass@k.
- This rule was re-confirmed after the 2026-06-23 30k loop0 confusion: full-data SI uniform Heun200 default p@10 is 50.0 by the primary all-candidate metric, not 52.0 valid-only.

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
- MLIP Pass headline reporting for current experiments uses all generated candidates as denominator; invalid candidates count as failures.
- Formula:
  - `c_s = |{X in S_hat_s : X is geometrically valid and E_theta(X; c) - E*_{theta,s} <= eps_succ}|`
  - `MLIP Pass@k(s) = 1 - C(n_s - c_s, k) / C(n_s, k)`
  - current OOD50 protocol uses `n_s = 100`, not `n_valid`.
  - with edge cases `c_s=0 -> 0`, `n_s-c_s < k -> 1`.
- Valid-only pass@k may be computed as a diagnostic under `valid_mlip_pass@k`, but it is not the headline number.

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
