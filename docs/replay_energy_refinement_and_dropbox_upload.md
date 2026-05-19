# Replay Energy Refinement and Dropbox Upload Summary

Date: 2026-05-20 KST  
Branch: `replay-energy-refinement`

## Codebase Changes

This branch adds the replay energy-refinement workflow used for the
`H200_ads_pair_dist_loss` run.

Main changes relative to `main`:

- Added UMA/OC20 energy reference generation:
  - `scripts/prepare_replay_energy_refs.py`
  - `scripts/compute_e_sys.py`
  - `scripts/run_compute_e_sys_8gpu.sh`
  - `scripts/merge_e_sys_and_rebuild_gt.py`
- Added reconvergence for previously unconverged `E_sys` samples:
  - `scripts/reconverge_e_sys.py`
  - `scripts/run_reconverge_8gpu.sh`
  - `scripts/finalize_reconverge_replay5000.sh`
  - `scripts/report_e_sys_steps.py`
- Updated replay success criteria to use OC20-scale MLIP references:
  - `adsorbgen/eval_replay.py`
  - `scripts/replay_daemon.py`
  - Success comparison now prefers per-system mean GT energy
    (`E_sys_mean`) when available.
  - The final replay workflow uses `success_margin=0.0`.
- Added derived MLIP-relaxed training LMDB support:
  - `scripts/build_mlip_relaxed_lmdbs.py`
  - Original LMDBs are left unchanged.
  - Converged MLIP-relaxed structures can replace `pos_relaxed` only in
    derived dataloader paths.
- Updated validation sampling for OC20-Dense:
  - `adsorbgen/dataset.py`
  - `adsorbgen/train.py`
  - Dense validation now supports unique sampling by `system_key`; the local
    OC20-Dense LMDB has 973 unique clean systems.
- Added external replay stream/daemon tooling and reports:
  - `scripts/replay_daemon.py`
  - `scripts/merge_replay_shards.py`
  - `scripts/report_replay_cycle.py`
  - `adsorbgen/tests/test_replay_stream.py`
- Added trajectory/export/diagnostic utilities:
  - `scripts/export_*`
  - `scripts/analyze_overlap_samples.py`
  - `scripts/success_rmsd_stats.py`
  - `scripts/trajectory_viewer_app.py`
  - `notebooks/flow_overlap_trajectory_viewer.ipynb`

## Current Runtime Workflow

The current long-running workflow is:

1. Re-relax non-converged `E_sys` shard records with:
   - model: `uma-s-1p1`
   - task: `oc20`
   - `fmax=0.05 eV/A`
   - `max_steps=300` additional FIRE steps after the first-pass 200-step run
     (about 500 total steps for previously unconverged samples)
2. Rebuild merged references:
   - `/home/irteam/data/replay/E_sys.pkl`
   - `/home/irteam/data/replay/gt_index_by_sid_oc20.pkl`
   - `/home/irteam/data/replay/gt_index_by_system_oc20.pkl`
3. Write convergence-step statistics:
   - `/home/irteam/data/replay/E_sys_step_stats.json`
4. Stop. No 5000-system x 10-placement replay cycle is launched by the
   finalize watcher.

Optional derived MLIP-relaxed LMDBs can be built separately by setting
`BUILD_MLIP_LMDBS=1` when running the finalize script. By default this is off.
For the current extra-`max_steps=300` reconvergence run, the active finalize
watcher was launched with `BUILD_MLIP_LMDBS=1`, so it will create
`/home/irteam/data/processed_mlip_oc20/*.lmdb` after rebuilding the final refs.

## Dropbox Upload

An rclone remote named `dropbox:` is configured on this machine.

The required-artifact uploader is:

```bash
/home/irteam/AdsorbGen/scripts/upload_required_after_reconverge.sh
```

It is currently launched as a background watcher. It waits until reconvergence
finishes and the merged reference files are rebuilt after launch time, so it
does not upload the older first-pass `E_sys.pkl` by mistake.

Upload destination:

```text
dropbox:AdsorbGen/H200_ads_pair_dist_loss_required_after_reconverge
```

Upload log:

```bash
/home/irteam/data/replay/dropbox_required_upload.log
```

### Required Files Uploaded

Replay reference artifacts:

```text
/home/irteam/data/replay/E_gas_only.pkl
/home/irteam/data/replay/E_slab_only.pkl
/home/irteam/data/replay/E_sys.pkl
/home/irteam/data/replay/gt_index_by_sid_oc20.pkl
/home/irteam/data/replay/gt_index_by_system_oc20.pkl
/home/irteam/data/replay/E_sys_step_stats.json
```

Logs:

```text
/home/irteam/data/replay/e_sys_finalize.log
/home/irteam/data/replay/finalize_reconverge_replay5000_*.log
/home/irteam/data/replay/finalize_reconverge_refs_*.log
/home/irteam/data/replay/e_sys_logs/
/home/irteam/data/replay/reconverge_logs/
/home/irteam/data/replay/dropbox_required_upload.log
```

Run outputs:

```text
/home/irteam/runs/H200_ads_pair_dist_loss/args.json
/home/irteam/runs/H200_ads_pair_dist_loss/train.log
/home/irteam/runs/H200_ads_pair_dist_loss/overlap_diag_300.json
/home/irteam/runs/H200_ads_pair_dist_loss/replay_stream/logs/
/home/irteam/runs/H200_ads_pair_dist_loss/replay_stream/shard_*/
/home/irteam/runs/H200_ads_pair_dist_loss/success_trajectories/
```

### Explicitly Not Uploaded

The upload intentionally excludes large or optional artifacts:

```text
/home/irteam/data/replay/e_sys_shards/
/home/irteam/data/replay/e_sys_shards_pre_reconverge/
/home/irteam/data/processed/
/home/irteam/data/processed_mlip_oc20/
```

It also excludes credentials and rclone config files.

## Manual Push

After the local commit is ready, push this branch with:

```bash
cd /home/irteam/AdsorbGen
git push origin replay-energy-refinement
```
