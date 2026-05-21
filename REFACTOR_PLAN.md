# AdsorbGen Refactor Plan

Drafted: 2026-05-21

Goal: make AdsorbGen understandable like AtomMOF: core package code is organized by domain, executable scripts are thin entry points, one-off experiment code is isolated, and active training/replay paths are easy to find.

## Phase 1 Status

Implemented on 2026-05-21:

- Root `scripts/` is now grouped into:
  - `scripts/train/`
  - `scripts/preprocess/`
  - `scripts/replay/`
  - `scripts/replay/e_sys/`
  - `scripts/analysis/`
- OC20-Dense and MLIP Pass one-off scripts moved into:
  - `experiments/2026-05-oc20dense/`
  - `experiments/2026-05-mlip-pass/`
- Older architecture search, anomaly audit, and OC20-Dense inspection helpers
  were deleted after deciding not to carry legacy-only code forward.
- `adsorbgen/scripts/` was reduced to active preprocessing CLIs only:
  - `preprocess_is2res.py`
  - `preprocess_oc20dense.py`
  - `unwrap_preprocess.py`

The remaining major work is Phase 3+: splitting large modules further and
introducing configs for repeatable train/replay/data workflows.

## Phase 2 Partial Status

Implemented on 2026-05-21:

- Created domain packages:
  - `adsorbgen/data/`
  - `adsorbgen/models/`
  - `adsorbgen/flow/`
  - `adsorbgen/evaluation/`
  - `adsorbgen/replay/`
- Moved core modules:
  - `dataset.py`, `multiplace.py`, `pbc_unwrap.py` -> `adsorbgen/data/`
  - `model.py`, `model_v2.py`, `model_factory.py`, `transformer.py`, `variants.py` -> `adsorbgen/models/`
  - `flow.py` -> `adsorbgen/flow/matching.py`
  - `eval.py`, `energy.py` -> `adsorbgen/evaluation/`
  - `replay.py`, `eval_replay.py`, `replay_viz.py` -> `adsorbgen/replay/`
- Updated active code, tests, experiments, and baseline helpers to import the
  new subpackage paths directly.
- Removed old top-level compatibility wrappers such as `adsorbgen/model.py`,
  `adsorbgen/dataset.py`, `adsorbgen/eval_replay.py`, and `adsorbgen/train.py`.

Still pending:

- Further split `adsorbgen/flow/matching.py` into loss/interpolation/sampling files.
- Further split `adsorbgen/training/train_cli.py` into module/datamodule/config files.
- Further split `adsorbgen/inference/cli.py` into checkpoint/model-loading helpers and sampling CLI.
- Add config files for common train/replay/data workflows if we want fewer
  shell flags.

## Current Problem

The repo currently mixes several layers:

- core library modules directly under `adsorbgen/`
- Python CLIs under `adsorbgen/scripts/`
- operational one-off Python scripts and shell launchers under root `scripts/`
- experiment-specific replay/MLIP pass/reconvergence code mixed with reusable code
- some legacy analysis helpers remain only where they still inspect active run
  artifacts

This makes it hard to answer simple questions like "what code is used for training?" or "which scripts are safe to ignore?"

## Refactor Principles

1. Keep the package name `adsorbgen`; do not switch to `src/` layout unless we also add packaging metadata.
2. Separate reusable library code from executable command-line scripts.
3. Keep shell launchers thin and put real logic in importable Python modules.
4. Keep one-off experiment scripts outside the active package path.
5. Move files in phases, then remove temporary compatibility wrappers once
   active imports are updated and tests pass.
6. Delete only after tests and one dry-run command pass.

## Proposed Target Layout

```text
AdsorbGen/
├── adsorbgen/
│   ├── data/
│   │   ├── datasets.py
│   │   ├── placement.py
│   │   ├── collate.py
│   │   └── pbc_unwrap.py
│   ├── models/
│   │   ├── dit.py
│   │   ├── dit_v2.py
│   │   ├── transformer.py
│   │   ├── factory.py
│   │   └── variants.py
│   ├── flow/
│   │   ├── matching.py
│   │   └── sampling.py
│   ├── training/
│   │   ├── module.py
│   │   ├── datamodule.py
│   │   ├── config.py
│   │   └── train_cli.py
│   ├── inference/
│   │   └── predict_cli.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── anomaly.py
│   │   ├── energy.py
│   │   └── mlip_pass.py
│   ├── replay/
│   │   ├── buffer.py
│   │   ├── eval.py
│   │   ├── stream.py
│   │   ├── daemon.py
│   │   └── viz.py
│   ├── preprocess/
│   │   ├── is2res.py
│   │   ├── oc20dense.py
│   │   ├── unwrap_center.py
│   │   └── mlip_targets.py
│   └── utils/
│       ├── lmdb.py
│       ├── paths.py
│       └── logging.py
├── configs/
│   ├── train/
│   ├── replay/
│   └── data/
├── scripts/
│   ├── train/
│   ├── replay/
│   ├── preprocess/
│   └── analysis/
├── experiments/
│   ├── 2026-05-replay/
│   └── 2026-05-mlip-pass/
├── baselines/
├── viz/
└── README.md
```

Notes:

- `scripts/` should mostly contain `.sh` launchers and tiny Python wrappers.
- Active Python implementation should live under `adsorbgen/`.
- Historical, paper-specific, or one-off scripts should live under `experiments/`.

## Current File Classification

### Core Library: Keep, But Move Into Subpackages

Current path | Target path | Notes
--- | --- | ---
`adsorbgen/data/dataset.py` | Later split into `datasets.py`, `placement.py`, `collate.py` | Old top-level wrapper removed.
`adsorbgen/data/pbc_unwrap.py` | Maybe later move to `adsorbgen/preprocess/unwrap_center.py` | Used by unwrap preprocessing.
`adsorbgen/flow/matching.py` | Later split into `losses.py`, `interpolation.py`, `sampling.py` | `adsorbgen.flow` package export keeps old imports working.
`adsorbgen/models/dit.py` | Done | Current main v1 model; old top-level wrapper removed.
`adsorbgen/models/dit_v2.py` | Done | Experimental v2 model; old top-level wrapper removed.
`adsorbgen/models/transformer.py` | Done | Model building blocks.
`adsorbgen/models/factory.py` | Done | Thin factory.
`adsorbgen/models/variants.py` | Done | Variant registry.
`adsorbgen/evaluation/metrics.py` | Later split anomaly into separate file | Old top-level wrapper removed.
`adsorbgen/evaluation/energy.py` | Done | UMA energy/relax interfaces; old top-level wrapper removed.
`adsorbgen/replay/buffer.py` | Later split stream classes if needed | Wrapper/package export remains at `adsorbgen/replay`.
`adsorbgen/replay/eval.py` | Done | Replay evaluation logic; old top-level wrapper removed.
`adsorbgen/replay/viz.py` | Done | Visualization helpers.
`adsorbgen/data/multiplace.py` | Done | Dataset-level sampling helper; old top-level wrapper removed.
`adsorbgen/training/train_cli.py` | Later split into `module.py`, `datamodule.py`, `config.py` | Run with `python -m adsorbgen.training.train_cli`.
`adsorbgen/inference/cli.py` | Later split into `predict_cli.py`, `checkpoint.py` | Package keeps `python -m adsorbgen.inference` working.

Temporary compatibility wrappers were removed after imports were updated.

### Active Preprocessing / Data Build Scripts

Current path | Target path | Keep?
--- | --- | ---
`adsorbgen/scripts/preprocess_is2res.py` | `adsorbgen/preprocess/is2res.py` plus CLI wrapper | Yes.
`adsorbgen/scripts/preprocess_oc20dense.py` | `adsorbgen/preprocess/oc20dense.py` plus CLI wrapper | Yes.
`adsorbgen/scripts/unwrap_preprocess.py` | `adsorbgen/preprocess/unwrap_center.py` plus CLI wrapper | Yes.
`scripts/preprocess/build_mlip_relaxed_lmdbs.py` | `adsorbgen/preprocess/mlip_targets.py` | Yes, current MLIP target workflow.
`scripts/preprocess/prepare_replay_energy_refs.py` | `adsorbgen/replay/energy_refs.py` or `adsorbgen/preprocess/energy_refs.py` | Keep if replay still uses it.
`experiments/2026-05-oc20dense/materialize_oc20dense_slab_refs.py` | `adsorbgen/preprocess/slab_refs.py` if reused | Currently experiment artifact.

### Active Replay / Self-Improvement

Current path | Target path | Keep?
--- | --- | ---
`scripts/replay/replay_daemon.py` | `adsorbgen/replay/daemon.py` plus shell wrapper | Yes.
`scripts/replay/run_replay_5000x10_8gpu.sh` | Keep in `scripts/replay/` | Active launcher.
`scripts/replay/report_replay_5000x10.py` | Keep in `scripts/replay/` or move to experiment later | Report utility.
`scripts/replay/e_sys/compute_e_sys.py` | `adsorbgen/replay/compute_e_sys.py` later | Keep if E_sys may be regenerated.
`scripts/replay/e_sys/reconverge_e_sys.py` | Keep in `scripts/replay/e_sys/` | Historical but useful audit.
`scripts/replay/e_sys/merge_e_sys_and_rebuild_gt.py` | Keep in `scripts/replay/e_sys/` | Reference generation.
`scripts/replay/e_sys/report_e_sys_steps.py` | Keep in `scripts/replay/e_sys/` | Analysis only.

### OC20-Dense / MLIP Pass One-Off Experiment

Current path | Target path | Keep?
--- | --- | ---
`experiments/2026-05-oc20dense/compute_oc20dense_mlip_relax.py` | Repro artifact | Keep.
`experiments/2026-05-oc20dense/merge_oc20dense_mlip_relax.py` | Repro artifact | Keep.
`experiments/2026-05-oc20dense/monitor_oc20dense_mlip_relax.py` | Repro artifact | Keep optional.
`experiments/2026-05-oc20dense/run_oc20dense_mlip_relax_4gpu.sh` | Repro artifact | Keep.
`experiments/2026-05-mlip-pass/eval_mlip_pass_lbfgs_ood50.py` | Repro artifact | Keep.
`experiments/2026-05-mlip-pass/merge_mlip_pass_lbfgs_ood50.py` | Repro artifact | Keep.
`experiments/2026-05-mlip-pass/run_mlip_pass_lbfgs_ood50_8gpu.sh` | Repro artifact | Keep.

### Legacy / Analysis Utilities

Current path | Target path | Keep?
--- | --- | ---
`scripts/analysis/compute_stats.py` | Analysis utility | Keep if useful.
`scripts/analysis/rescore_anomaly.py` | Analysis utility | Keep only if saved inference analysis is active.
`scripts/analysis/plot_anomaly.py` | Analysis utility | Keep only with rescore.

Legacy-only architecture search, anomaly audit, and OC20-Dense inspection
helpers were deleted instead of preserved as archive code.

## Recommended Refactor Phases

### Phase 0: Freeze Current State

- Ensure no training/replay process is running.
- Commit or stash current work.
- Save `previous_context.md`.
- Run smoke tests:

```bash
PYTHONPATH=/home/irteam/AdsorbGen pytest -q adsorbgen/tests
```

### Phase 1: Non-Destructive Organization

- Create new directories.
- Move only one-off experiment scripts out of the active package path.
- Keep shell launchers under `scripts/`.
- Add short README files in:
  - `scripts/`
  - `experiments/`
  - `adsorbgen/`

Risk: low, if moved scripts are not imported.

### Phase 2: Split Core Package

- Move core modules into subpackages.
- Update imports to the new subpackage paths.
- Delete old top-level wrappers after tests pass.

Risk: moderate. Requires tests after every small batch of moves.

### Phase 3: Split Training CLI Further

Break `adsorbgen/training/train_cli.py` into:

- `adsorbgen/training/module.py`
- `adsorbgen/training/datamodule.py`
- `adsorbgen/training/config.py`
- `adsorbgen/training/train_cli.py`

Run training with:

```bash
python -m adsorbgen.training.train_cli
```

Risk: high. Do after package split is stable.

### Phase 4: Add Configs

Introduce `configs/` for common runs:

- `configs/train/h200_pairdist.yaml`
- `configs/train/b_center_rel_pairdist_id.yaml`
- `configs/train/b1_ads_only_center_rel_pairdist_id.yaml`
- `configs/replay/5000x10.yaml`
- `configs/data/processed_mlip_oc20.yaml`

This reduces huge shell command lines and makes runs easier to inspect.

### Phase 5: Finish Cleanup

Only after commands and tests pass:

- split any remaining large modules if they keep obscuring the active path
- delete remaining wrappers only if current commands and external launchers no
  longer need old import paths

## Minimal First Refactor I Recommend

Do this first:

1. Create:

```text
experiments/2026-05-replay/
experiments/2026-05-oc20dense/
experiments/2026-05-mlip-pass/
scripts/train/
scripts/replay/
scripts/preprocess/
scripts/analysis/
```

2. Move one-off root scripts into `experiments/`.
3. Move shell launchers into `scripts/{train,replay,preprocess}`.
4. Keep importable active package code unchanged.
5. Add `scripts/README.md` and `experiments/README.md`.

This gives immediate readability without touching model/training imports.

## Open Questions Before Actually Moving Files

1. Should we preserve old command paths through wrapper scripts, or is it okay to update all commands now?
2. Should `previous_context.md` remain at repo root, or move into `experiments/2026-05-replay/NOTES.md`?
3. Should aborted run directories from the accidental launch be deleted?
4. Should architecture-search utilities be archived or deleted?
5. Do we want Hydra-style configs like AtomMOF, or a simpler YAML + argparse loader?

## Verification

After the Phase 1/2 cleanup, the stale tests that depended on removed
displacement-flow APIs were rewritten for the current absolute-coordinate API.

```bash
PYTHONPATH=/home/irteam/AdsorbGen /home1/irteam/micromamba/envs/adsorbgen/bin/python -m pytest -q adsorbgen/tests
```

Result: `37 passed, 3 skipped`.
