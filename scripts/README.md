# Scripts

This directory now contains thin operational entry points grouped by workflow.
Reusable training, model, data, replay, and evaluation logic should live under
`adsorbgen/`; one-off research reproductions should live under `experiments/`.
The core package is now partially organized into `adsorbgen/data`,
`adsorbgen/models`, `adsorbgen/evaluation`, and `adsorbgen/replay`.

## Layout

- `train/`: training launchers.
- `preprocess/`: data/reference preparation commands.
- `replay/`: replay/self-improvement launchers and replay daemon.
- `replay/e_sys/`: MLIP `E_sys` reference generation and reconvergence tools.
- `analysis/`: analysis utilities that inspect existing runs or LMDBs.

## Active Entry Points

- `scripts/train/launch_with_external_replay.sh`
- `scripts/preprocess/build_mlip_relaxed_lmdbs.py`
- `scripts/replay/replay_daemon.py`
- `scripts/replay/run_replay_5000x10_8gpu.sh`
- `scripts/replay/e_sys/run_compute_e_sys_8gpu.sh`
- `scripts/replay/e_sys/run_reconverge_8gpu.sh`

One-off scripts from the May 2026 OC20-Dense/MLIP Pass work are under
`experiments/`; older archive-only utilities were deleted instead of carried
forward.
