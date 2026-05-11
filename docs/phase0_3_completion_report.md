# Phase 0–3 Completion Report

## Phase 0 — Preflight (all done)

| Task | Status | Artefact |
|---|---|---|
| 0a Phase3 flag verification | ✓ | Finding: phase3 re-relaxes from DFT-relaxed, so 4/5 flags trivially False. Check functions themselves work. Rerun NOT needed. |
| 0b OC20-Dense merge | ✓ | `data/processed/oc20dense.lmdb` (65,073 entries + metadata) |
| 0c GT replay index | ✓ | `data/replay/gt_index_by_{system,sid}.pkl` (489,453 / 529,266 systems eligible = 92.5%) |
| 0d Placement preflight | ✓ | 500/500 placements succeeded → runtime try/except, no prefilter |

## Phase 1 — Flow framework refactor (all done)

Files modified:
- `adsorbgen/flow.py` — rewrote as placement-prior flow. Deleted `sigma`, `sample_delta0`, MIC-in-loss. `corrupt` now `delta_t = t·delta_1`. `euler_sample` starts from zeros.
- `adsorbgen/dataset.py` — added `PlacementPriorDataset` (fresh fairchem placement per `__getitem__`) + `MixedReplayDataset`.
- `adsorbgen/model.py` — deleted `DeltaEEmbedder`, `use_delta_e`, `delta_e_max`, `delta_e_freq_dim`, `sigma`. Removed `delta_e` / `cond_drop` forward kwargs.
- `adsorbgen/model_v2.py` — deleted stale `sigma` field.
- `adsorbgen/multiplace.py` — added `prior_mode` + `interstitial_gap` params, 3-mode dispatch.
- `adsorbgen/inference.py` — removed sigma / delta_e / cond_drop; added `--prior-mode` CLI.
- `adsorbgen/train.py` — swapped to `PlacementPriorDataset`; added `--prior-mode`, `--interstitial-gap`; `compute_delta1`/`corrupt` signature updated.
- `adsorbgen/variants.py` — removed `v1-dec-pair-no-de` variant (ΔE ablation no longer meaningful).

Phase 1 smoke:
- All 9 modules import clean
- Batch of 4 from `is2res_train`: ads diff 5.6–9.0 Å (placement active)
- Forward loss 2.15, grads finite
- Euler sample runs end-to-end

## Phase 2 — Replay infrastructure (all done)

Files added:
- `adsorbgen/replay.py` — `ReplayBuffer` (append/replace modes, per-system cap, global cap, weighted sampling), `ReplayEntry`. 5 unit tests pass (per-system cap, weighted sample, replace mode, global cap, save/load).
- `adsorbgen/energy.py` — added `UMARelaxer` (LBFGS via FAIRChemCalculator, same logic as phase3).
- `adsorbgen/eval_replay.py` — `run_replay_eval` orchestrator, `ReplayScheduler` (plateau / 5-eval trigger).
- `adsorbgen/dataset.py` — `MixedReplayDataset` (α-biased base↔replay sampling).

## Phase 3 — End-to-end smoke (all done, CUDA_VISIBLE_DEVICES=0)

`scripts/phase3_smoke.py` executed successfully:
- 40-step mini training on GPU 0 with MixedReplayDataset (α=0.5), tiny v1-wide-no-gate (1 enc + 2 trunk + 1 dec blocks, atom_s=token_s=64)
- Final loss 4.68, no NaN, 66 ms/step on GPU 0
- Buffer save/load roundtrip verified
- **2-rank gloo DDP buffer sync verified** (rank 0 writes, both read same canonical file)

**GPU constraint note**: Only GPU 0 was idle at run time; others 98-100%. Model/batch sizes shrunk accordingly. No OOM.

## Artefacts summary

```
/home/minkyu/Cat-bench/
├── AdsorbGen/
│   ├── adsorbgen/
│   │   ├── flow.py              [rewrite]
│   │   ├── dataset.py           [+PlacementPriorDataset, +MixedReplayDataset]
│   │   ├── model.py             [ΔE stripped]
│   │   ├── model_v2.py          [sigma stripped]
│   │   ├── multiplace.py        [+prior_mode, +interstitial_gap]
│   │   ├── inference.py         [ΔE/sigma stripped, +prior-mode CLI]
│   │   ├── train.py             [+PlacementPriorDataset, +prior-mode CLI]
│   │   ├── variants.py          [v1-dec-pair-no-de removed]
│   │   ├── energy.py            [+UMARelaxer]
│   │   ├── replay.py            [NEW]
│   │   └── eval_replay.py       [NEW]
│   └── docs/
│       ├── abs_coord_flow_plan.md
│       ├── phase0_preflight_report.md
│       └── phase0_3_completion_report.md  ← this file
├── data/
│   ├── processed/
│   │   └── oc20dense.lmdb                             [merged dense set]
│   └── replay/
│       ├── gt_index_by_system.pkl
│       └── gt_index_by_sid.pkl
└── scripts/
    ├── merge_oc20dense_test.py
    ├── build_replay_gt_index.py
    ├── preflight_placement.py
    ├── preflight_phase3_flags.py
    └── phase3_smoke.py
```

## Next steps — Phase 4 (awaits user approval)

Production training run. Needs:
- Full-data IS2RES (~535k samples) training
- DDP across N GPUs (GPU availability dependent)
- Real UMA relaxation in eval_replay every 30 epochs
- W&B logging, checkpoint save, buffer persist per eval

**Recommended launch command** (skeleton — user approves + adjusts GPUs):
```bash
PYTHONPATH=AdsorbGen python -m adsorbgen.train \
    --arch v1 --variant v1-wide-no-gate \
    --prior-mode random_heuristic \
    --train-lmdb data/processed/is2res_train.lmdb \
                 data/processed/is2res_val.lmdb \
                 data/processed/is2res_val_ood_ads.lmdb \
                 data/processed/is2res_val_ood_cat.lmdb \
                 data/processed/is2res_val_ood_both.lmdb \
    --val-lmdb   data/processed/oc20dense.lmdb \
    --out runs/v1-wide-no-gate-abs \
    --epochs 300 --batch-size 8 --devices 4 \
    --lr 1e-4 --loss-type l1 \
    --wandb-project adsorbgen-abs --wandb-run-name v1wng-abs
```

(Replay integration in train.py CLI wires to be added in Phase 4 launch; the Python API is ready.)
