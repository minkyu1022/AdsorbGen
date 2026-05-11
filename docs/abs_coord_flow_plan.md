# Plan: Placement-Prior Flow Matching (full replacement)

**Status**: v2 (decisions locked)
**Scope**: replace Gaussian-noise displacement flow with structured-prior displacement flow where `x_0` (= current model's `pos`) is a fresh fairchem adsorbate placement drawn every training iteration. Full rip-out of old flow — no backwards compat, no ΔE conditioning.

---

## 1. Core insight (simplifies everything)

In AtomMOF / current model.py convention, the forward signature is `(pos, delta_t, t, ...)` with `x_t = pos + delta_t`. The `pos` channel **is** the reference / prior position.

Under the new framework, we redefine what `pos` means:
- **Old**: `pos` = LMDB's stored initial structure (with its original fairchem placement)
- **New**: `pos` = freshly drawn x_0 each iter, i.e. `[surface: LMDB pos_init, ads: fresh fairchem placement, bulk: LMDB pos_init]`

Everything downstream is mechanically unchanged:
- `delta_t = (1-t)·delta_0 + t·delta_1` — but now `delta_0 = 0` (no Gaussian noise; x_0 is structured)
- `delta_1 = pos_relaxed - pos` — raw Cartesian, **no MIC in loss** (decision #11)
- Model predicts `delta_1`; output head zero-init stays valid (small `delta_1` at init = sane output)
- Inference: start at `delta_t = 0`, integrate, output `x_out = pos + delta_{t=1-eps}`

**Model architecture: zero changes.** Only the data pipeline, flow config, and delta_0 path change.

---

## 2. Locked decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Prior mode | Config-selectable: `"random"` or `"random_heuristic"` (default) |
| 2 | ΔE conditioning | **Remove** (`use_delta_e=False` deleted from config; DeltaEEmbedder class + code path ripped out) |
| 3 | Baseline arch | v1-wide-no-gate |
| 4 | Backwards compat | **None** — displacement flow code deleted, not dual |
| 5 | x_0 as separate model input | Not needed — `x_0 = pos` already. No new proj added. |
| 6 | Surface prior | Deterministic LMDB pos_init (no jitter) |
| 7 | Loss weighting | Equal per movable atom initially; reassess after first run based on per-group MAE |
| 8 | Placement freshness | Live fairchem call every iteration (num_sites=1) |
| 9 | Dual validation | Skipped |
| 10 | Data splits | Training = **IS2RES** (train 460k + val 25k + 3 OOD vals ≈ 535k). Test = **OC20-Dense** (train+val merged = 65,077) held out for final global-min coverage evaluation only. No arch-search/final split — single pipeline. |
| 11 | MIC in loss | **Never**. MIC stays only in pair-feature distance computation (as today) |
| A | LMDB `pos` for ads | Discarded — re-sampled per iter |
| B | Pristine slab vs LMDB pos[slab_idx] | Pristine (adsorbate-free slab). In practice equal, but semantics are pristine |

---

## 3. Math summary

```
x_0     = [surface_idx: LMDB pos_init, ads_idx: fairchem placement, bulk_idx: LMDB pos_init]
x_1     = LMDB pos_relaxed
delta_1 = (x_1 - x_0)  * movable_mask                      # raw Cartesian, no MIC
delta_0 = 0                                                 # structured prior removes need for Gaussian
delta_t = t · delta_1                                       # simplified linear path
x_t     = x_0 + delta_t                                     # internal to model via xt_proj(pos + delta_t)

Loss  = || model(pos=x_0, delta_t, t, ...) - delta_1 ||^2   on movable atoms (same reduction as today)

Euler (inference):
    delta_t <- 0
    for i in 0..num_steps-1:
        pred_delta_1 = model(pos=x_0, delta_t, t_i, ...)
        v            = (pred_delta_1 - delta_t) / (1 - t_i)
        delta_t     += dt * v                   # (SDE variant keeps 0.5*g^2*score + noise term)
    x_out = x_0 + delta_t
```

---

## 4. Files to change

### 4.1 `adsorbgen/flow.py` — simplify

Delete / edit:
- `sigma` field from `FlowConfig` (no Gaussian prior anymore)
- `sample_delta0()` — delete (always zero)
- `corrupt()` — simplify to `delta_t = t·delta_1` (no delta_0 sampling)
- `compute_delta1()` — **remove MIC call**. `delta_1 = (pos_relaxed - pos) * movable_mask`. No `minimum_image()`.
- `euler_sample()` — start from `delta_t = 0` (zeros tensor) instead of `sample_delta0(...)`. SDE/refine/FK paths unchanged.

Keep:
- `minimum_image()` utility (used by pair feature construction in model.py / model_v2.py)
- FK steering, refine_final, SDE logic — all orthogonal

### 4.2 `adsorbgen/dataset.py` — new dataset class

Add `PlacementPriorDataset`. Replaces `PreprocessedDisplacementDataset` as the training-time dataset. The collate function returns the same keys except `pos` now contains the fresh prior sample instead of the LMDB original.

Key implementation points:
- Per-worker lazy load of `adsorbates.pkl` (identical to multiplace.py)
- Per-worker per-sample fairchem call using `mode=self.prior_mode` (one of `"random"` / `"random_site_heuristic_placement"`)
- Fallback: if fairchem returns zero sites (rare — tiny slabs), raise — we don't silently default
- Preserves existing keys: `pos_relaxed`, `tags`, `atomic_numbers`, `movable_mask`, `fixed`, `cell`, `sid`, `ads_id`
- `pos` key: fresh x_0 every call. Adsorbate positions from fairchem; non-adsorbate atoms take LMDB's pos_init (pristine slab per decision B)

Validation dataset: also uses `PlacementPriorDataset` with same config. At val/search time we want placement-consistent priors.

Retire `PreprocessedDisplacementDataset` after migration or rename it `LegacyFixedPlacementDataset` if still useful for one-off debugging.

### 4.3 `adsorbgen/train.py`

- Swap dataset class (`PreprocessedDisplacementDataset` → `PlacementPriorDataset`)
- Pass `prior_mode` from CLI / variant config
- `training_step`:
  - `delta_1 = compute_delta1(pos, pos_relaxed, cell, movable)` — using the updated (non-MIC) compute
  - `delta_t = corrupt(delta_1, t, cfg, movable)` — returns `t·delta_1`
  - model call: same as today
  - loss: same
- `validation_step` / `sample_eval`: same change (new pos = prior sample)

### 4.4 `adsorbgen/model.py`

- Delete `DeltaEEmbedder` class
- Delete `delta_e_embedder` attribute and `use_delta_e` config field
- Delete `cond_drop` logic and `e_cond` add in forward
- Forward signature: drop `delta_e`, `cond_drop` arguments
- Everything else (encoder/trunk/decoder, skip_stage_gates, pair features, trunk_init stack) unchanged
- The CLI `--arch v1 --variant v1-wide-no-gate` flow still works

### 4.5 `adsorbgen/model_v2.py`

- V2 doesn't have ΔE anyway — no change needed here except consistency check
- Confirm forward signature matches model.py (no delta_e / cond_drop args)

### 4.6 `adsorbgen/inference.py` (search)

- Replace custom "start from LMDB pos + Gaussian noise" path with start from prior-sampled pos
- Reuse `MultiPlacementDataset` (already does K placements per system)
- Pass `prior_mode` config through
- Downstream search / ranking logic unchanged

### 4.7 `adsorbgen/multiplace.py`

- Add `prior_mode` parameter to `MultiPlacementDataset.__init__`, plumb through to `AdsorbateSlabConfig(mode=prior_mode)`
- Default stays `"random_site_heuristic_placement"` for backward match with current leaderboard

### 4.8 `adsorbgen/variants.py`

- Add `prior_mode` as a supported variant field (default `"random_heuristic"`)
- Strip `use_delta_e` from variant definitions (was `v1-dec-pair-no-de`, etc.) — it's no longer a knob
- Mark variants that required ΔE as removed / deprecated

### 4.9 Tests

- `tests/test_placement_prior.py`: dataset returns fresh pos each call; ads atoms differ, surface atoms stay at pos_init
- `tests/test_abs_flow.py`: `delta_0 = 0`, `delta_t = t·delta_1`, euler sample from zero start
- `tests/test_no_mic_in_loss.py`: `compute_delta1` is a plain Cartesian subtract
- Existing tests: delete anything that touches sigma, delta_0 sampling, or ΔE conditioning

---

## 5. Concrete code diffs (representative)

### flow.py — compute_delta1
```python
# before
def compute_delta1(pos, pos_relaxed, cell, movable_mask):
    d = minimum_image(pos_relaxed - pos, cell)   # <-- remove
    return d * movable_mask.unsqueeze(-1).to(d.dtype)

# after
def compute_delta1(pos, pos_relaxed, movable_mask):
    d = pos_relaxed - pos
    return d * movable_mask.unsqueeze(-1).to(d.dtype)
```

### flow.py — corrupt
```python
# before
def corrupt(delta1, t, cfg, movable_mask):
    delta0 = sample_delta0(...) * movable_mask...
    delta_t = (1 - t_b) * delta0 + t_b * delta1
    return delta_t, delta0

# after
def corrupt(delta1, t, movable_mask):
    t_b = t.view(-1, 1, 1).to(delta1.dtype)
    return t_b * delta1      # delta_0 = 0
```

### flow.py — euler_sample init
```python
# before: delta_t = sample_delta0(...) * movable_f
# after:  delta_t = torch.zeros_like(pos) * movable_f   # explicit shape, ensures pad-safe
```

### dataset.py — PlacementPriorDataset.__getitem__ sketch
```python
def __getitem__(self, i):
    entry = self.base[i]                           # raw LMDB read
    if entry["ads_id"].item() < 0:
        raise RuntimeError(f"sample {i}: ads_id=-1 (not supported)")
    ads_mask = (entry["tags"] == 2).numpy()
    slab_mask = ~ads_mask
    # pristine slab = LMDB pos[slab_idx] (bulk + surface, never modified)
    pristine_slab_pos = entry["pos"].numpy()[slab_mask]
    placement_pos = self._place(entry, pristine_slab_pos)   # fairchem call, (n_ads, 3)
    new_pos = entry["pos"].clone()
    new_pos[ads_mask] = torch.from_numpy(placement_pos)
    entry["pos"] = new_pos
    return entry
```

---

## 6. Placement mode config

```python
# variants.py
"v1-wide-no-gate-rhp": {"prior_mode": "random_heuristic"}   # default, matches current
"v1-wide-no-gate-rnd": {"prior_mode": "random"}             # new ablation

# CLI
--prior-mode {random, random_heuristic}
```

Internal mapping (in `PlacementPriorDataset` and `MultiPlacementDataset`):
```python
_PRIOR_MODE_MAP = {
    "random":          "random",
    "random_heuristic": "random_site_heuristic_placement",
}
```

---

## 7. Training-time overhead estimate

- fairchem `AdsorbateSlabConfig(num_sites=1)` call: ~20-50ms (heuristic mode) / ~5-15ms (random mode)
- 65k samples/epoch × 30ms / 8 workers ≈ 4 min overhead/epoch
- 10-ep sweep → ~40 min added. Acceptable.
- If profile shows IO bottleneck, add per-worker cache of M=5-10 placements.

---

## 8. Validation strategy

1. **Smoke**: 1 epoch on 1k samples with `v1-wide-no-gate` + `prior_mode=random_heuristic`. Assert loss drops, no NaN. Log per-group (surface vs ads) MAE.
2. **10-ep sweep**: both prior modes on v1-wide-no-gate. Compare strict valid_rate vs current leaderboard's v1-wide-no-gate (0.836). Target: match or exceed.
3. **Loss-scale audit**: after smoke, inspect surface-only loss vs ads-only loss. If imbalance > 5x, switch to two-term loss (§2 #7).
4. **Final train**: full OC20-Dense train+val with anomaly filter per `project_catbench_final_eval_protocol.md`. Only after sweep passes.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| fairchem call fails on edge-case slabs (tiny cells, exotic symmetries) | Dataset raises; caller must filter data. Pre-scan LMDB to identify bad `ads_id` / slab combos before training. |
| Ads travels further (placement → relaxed ~2-5 Å) vs old setup (init → relaxed ~0.5 Å), so `delta_1` magnitude grows → loss grows | Expected. Adjust learning rate if needed. Log per-group loss to catch scale issues. |
| Random mode produces unphysical placements (atoms too close to surface / overlapping) | fairchem's `interstitial_gap=0.1` parameter handles this. Confirm `random` mode respects it. |
| Atom order drift between placement and relaxed | Already validated in multiplace.py — enforces `atomic_numbers[ads_idx] == canonical_nums` |
| Old checkpoints unloadable | Expected — user chose full replacement. Document in README. |

---

## 10. Deliverables checklist

- [ ] `adsorbgen/flow.py`: remove sigma/delta_0/MIC in loss; simplify corrupt, compute_delta1, euler_sample
- [ ] `adsorbgen/dataset.py`: add `PlacementPriorDataset`; retire displacement dataset
- [ ] `adsorbgen/model.py`: rip out DeltaEEmbedder + use_delta_e field + cond_drop path
- [ ] `adsorbgen/model_v2.py`: confirm no ΔE leftovers (already clean)
- [ ] `adsorbgen/multiplace.py`: accept `prior_mode` param
- [ ] `adsorbgen/inference.py`: plumb `prior_mode`; start integration from delta_t=0
- [ ] `adsorbgen/train.py`: swap dataset, update flow calls, drop ΔE args
- [ ] `adsorbgen/variants.py`: add `prior_mode`; delete ΔE-related variants
- [ ] tests: placement dataset, flow simplification, no-MIC-in-loss
- [ ] docs: update `AdsorbGen/modify_architecture.md` with the new formulation
- [ ] smoke + 10-ep sweep results logged before final-train commitment

---

## 11. fairchem constants

- `interstitial_gap = 0.1` Å (current multiplace default). Definition: minimum distance between adsorbate and slab covalent radii along surface normal at placement. Smaller = tighter binding site, larger = adsorbate floats higher. Kept unchanged between train-time and inference-time placement to ensure prior distribution match.
- `num_sites = 1`, `num_augmentations_per_site = 1` at train time (one fresh placement per iter).

## 12. Bad ads_id handling

Preflight the LMDB once, write a clean index listing samples with `ads_id >= 0` and successful fairchem call. Training dataset reads only from this index. Runtime raise if a sample still fails (should never happen post-preflight). Machinery: new `scripts/preflight_placement_prior.py`.

## 13. Prior mode (finalized)

Three modes supported, mapped to fairchem `AdsorbateSlabConfig`:
```python
_PRIOR_MODE_MAP = {
    "random":           "random",                          # pure uniform
    "heuristic":        "heuristic",                       # pymatgen AdsorbateSiteFinder
    "random_heuristic": "random_site_heuristic_placement", # Delaunay random (DEFAULT)
}
```

---

# Part B — Replay Buffer (Expert Iteration)

## 14. Rationale

Goal: search for global minima of adsorption energy across the enormous (placement × basin) landscape. DFT-provided relaxed structures are **local minima only**; different initial placements converge to different basins. Model inference with diverse priors can discover lower-energy basins than any GT placement for a given (slab, ads). Using these discoveries as new training targets creates a self-improvement loop toward global minima.

This is expert iteration (AlphaZero family) with an energy-based reward filter. User-verified intent.

## 15. GT reference energies (available)

`/home/minkyu/Cat-bench/results/phase3/adsorption_results.pkl` — 544,182 entries keyed by SID. Contains:
- `E_sys` (UMA-relaxed total energy of slab+ads)
- `E_slab` (UMA clean-slab energy; per-system constant, drops out of comparison)
- `E_ads_no_gas_ref = E_sys - E_slab`
- Pre-computed anomaly flags: `dissociated`, `desorbed`, `surface_changed`, `intercalated`, `pre_dissociated`
- `converged`, `forces_max`, `n_steps`

Comparison uses `E_sys`: `success ⟺ E_sys_pred + δ < E_sys_gt` (clean-slab term cancels).

GT filtering: only samples where GT `converged=True` and all anomaly flags `False` are eligible (otherwise the GT itself is bad and "improvement" is meaningless).

## 16. Locked replay params

| # | Parameter | Value |
|---|---|---|
| 1 | Inference mode | UMA **relaxation** (not single-point) to obtain predicted `E_sys_pred` |
| 2 | GT energy source | `phase3/adsorption_results.pkl` |
| 3 | Success margin δ | `replay_success_margin`, default **`0.05 eV`** (user-confirmed). CLI override available. |
| 4 | Replay ratio α | 0.5 (half batch from replay when buffer non-empty) |
| 5 | Eval frequency | every 30 epochs |
| 6 | Eval budget | 2000 systems × 5 placements per eval (= 10k candidates) |
| 7 | Buffer cap | `2 × |training_dataset|` = `2 × 535k ≈ 1.07M`. Configurable via `--replay-cap`. |
| 8 | Eviction (intra-system) | per-system cap 10; evict highest-`E_sys` within system |
| 9 | Eviction (global) | when buffer > cap, evict global-highest `E_sys` |
| 10 | Warmup | 30 epochs pure training before first eval |
| 11 | Anomaly filter | dissoc + desorb + surf_changed + intercalated + overlap<0.5Å |
| 12 | Replay sample weight | proportional to `improvement = E_gt - E_pred` |
| 13 | Per-system cap | 10 entries |
| 14 | Persistence | mandatory — buffer saved to disk, new-samples log saved, both reloadable |

## 17. Replay entry schema

```python
@dataclass
class ReplayEntry:
    system_key:   tuple                 # (sid, ads_id) as primary key
    x_1:          np.ndarray            # (N, 3) predicted relaxed coords (new target)
    tags:         np.ndarray            # (N,) for sanity check at replay time
    atomic_numbers: np.ndarray          # (N,) same reason
    E_sys_pred: float              # UMA E(slab+ads) after our relaxation
    E_sys_gt:   float              # from phase3 pkl
    improvement:     float              # E_gt - E_pred (> δ by construction)
    epoch_added:     int
    source_placement_mode: str          # "random" / "heuristic" / "random_heuristic"
```

`x_0` is NOT stored — at replay train time we draw a fresh placement for x_0, same as baseline training. The replay entry only provides an updated `x_1` target for that system.

## 18. Eval procedure (every 30 epochs)

```
for each (system, K=5 placements) in sampled 2000 systems:
    for k in range(K):
        x_0 = fairchem_placement(slab, ads_id, mode=prior_mode)
        x_1_pred = flow_sample(model, x_0)            # Euler integration
        atoms_pred = to_ase(x_1_pred, slab, ads)
        atoms_relaxed = UMA_relax(atoms_pred, steps=100, fmax=0.05)
        E_pred = UMA_single_point(atoms_relaxed)       # E_sys_pred
        if converged and passes_anomaly_filter(atoms_relaxed):
            if E_pred + δ < E_gt[system]:
                add_to_buffer(ReplayEntry(...))
```

UMA relaxation: reuse the existing Phase 3 relaxation logic (`scripts/phase3_adsorption.py`) via importable helper in `adsorbgen/energy.py`. Need to add batched-relaxation API (currently single-atom path). Budget estimate:
- 10k candidates × ~50 UMA steps each × ~0.1s/step ≈ **14 hours/eval** (hefty)
- Parallelize across K GPUs: ~3-4 hours with 4 GPUs
- If too slow: initial runs with 500 systems × 3 placements (~45 min/eval with 4 GPUs)

## 19. Training batch mixing

```python
class MixedReplayDataset:
    def __init__(self, base_dataset, replay_buffer, alpha):
        self.base = base_dataset
        self.buf = replay_buffer
        self.alpha = alpha

    def __len__(self):
        # define epoch size as len(base); replay samples resample every iter
        return len(self.base)

    def __getitem__(self, i):
        if len(self.buf) > 0 and random.random() < self.alpha:
            # weighted sample by improvement
            entry = self.buf.sample_weighted()
            x_0 = fairchem_placement(slab_from_entry, entry.ads_id, mode=prior_mode)
            x_1 = entry.x_1
        else:
            sample = self.base[i]
            x_0 = fairchem_placement(sample.slab, sample.ads_id, mode=prior_mode)
            x_1 = sample.pos_relaxed
        return {"x_0": x_0, "x_1": x_1, ...}
```

First 30 epochs: buffer empty, α effectively 0.

## 20. DDP & persistence

- Rank 0 owns the canonical buffer, lives in `runs/<variant>/replay_buffer.pkl`.
- Per-eval: each rank computes eval on its shard, reports candidates to rank 0 via `all_gather_object`. Rank 0 de-duplicates and writes.
- Before each epoch: all ranks read the canonical file (cheap; buffer < 1GB).
- Sample new-additions log: `runs/<variant>/replay_new_samples_ep{N}.pkl` (never overwritten), for post-hoc analysis.
- At training end: keep both files. Future runs can init buffer from the latest `replay_buffer.pkl`.

## 21. Metrics to log

- `replay/buffer_size`
- `replay/new_samples_this_eval`
- `replay/mean_improvement_this_eval`
- `replay/fraction_systems_covered` = num systems with any entry / total systems
- `replay/per_system_cap_saturation` = fraction of systems at the 10-cap
- Training-loss breakdown: `loss_base` vs `loss_replay` separately

## 22. Risks

| Risk | Mitigation |
|---|---|
| UMA reward hacking (predicts low-E unphysical structure UMA accepts) | Anomaly filter + margin δ; spot-check random new entries with DFT single-point periodically |
| Distribution shift (buffer dominates, DFT knowledge lost) | α=0.5 cap; monitor val MAE on original (non-replay) set |
| Eval compute blowup | Start small (500×3); scale up only if bottleneck is elsewhere |
| Per-system cap favors chemistry-easy systems | Per-system quota 10 guarantees minimum representation |
| Buffer growth unbounded | Hard cap 2×dataset; eviction on overflow |
| Stale replay targets (buffer from epoch 30 used in epoch 200) | Periodic re-eval overwrites if lower energy found (replace mode); append mode natural drift |
| DDP buffer desync | Rank 0 only writer; file-based canonical state |

## 23. Deliverables checklist (Part B)

- [ ] `adsorbgen/replay.py`: `ReplayEntry`, `ReplayBuffer` with append/replace modes, intra-system + global eviction, weighted sampling
- [ ] `adsorbgen/eval_replay.py`: eval loop (flow sample → UMA relax → filter → add)
- [ ] `adsorbgen/energy.py`: add batched UMA relaxation helper (reuse phase3 logic)
- [ ] `adsorbgen/dataset.py`: `MixedReplayDataset`
- [ ] `adsorbgen/train.py`: `--use-replay`, `--replay-mode`, `--replay-ratio`, `--replay-eval-every`, `--replay-success-margin`, `--replay-cap`, `--replay-eval-systems`, `--replay-eval-placements`, `--replay-warmup-epochs`
- [ ] `scripts/preflight_placement_prior.py`: drop bad ads_id / fairchem-incompatible slabs
- [ ] tests: replay buffer unit (eviction, weighted sampling, DDP merge); end-to-end mini-run (10 epochs warmup + 1 eval + 5 epochs replay)
- [ ] persistence: buffer saved every eval; new-additions logged; resume-ready

## 24. Naming & dataset clarifications (user-confirmed)

1. **`E_slab_ads` → `E_sys`** globally (done via sed in this plan).
2. **OC20-Dense**: train/val split collapsed. Merged file = single **test set** for post-training global-min coverage. Not touched during training.
3. **Training data**: IS2RES (train 460,330 + val 24,945 + 3 OOD vals ≈ 75k) ≈ **535k total**.
4. **No arch-search phase going forward**. v1-wide-no-gate is locked baseline. Replay is part of the single production training pipeline from epoch 0.

## 25. GT pre-filter stats (phase3/adsorption_results.pkl, 544,182 entries)

```
converged=True:           99.7%
no pre_dissociated:      100.0%   ← flag never True
no dissociated:          100.0%   ← flag never True
no desorbed:             100.0%   ← flag never True
no intercalated:         100.0%   ← flag never True
no surface_changed:       92.6%
ALL CLEAN:                92.4%
REJECTED:                  7.6%   (mostly surface_changed; tiny non-convergence)
```

**Flag gotcha**: four of five anomaly flags (pre_diss/diss/desorb/inter) are uniformly False across all 544k entries. Two possibilities:
- (A) The IS2RES data was already filtered by these criteria upstream (phase3 only stored passing samples). Then flags are authoritative False = clean.
- (B) `phase3_adsorption.py` didn't actually run DetectTrajAnomaly, just initialized flags to False. Then GT may contain unphysical configs we think are clean.

**Action**: spot-check by re-running DetectTrajAnomaly on 100 random phase3 entries. If any come back True, option (B) — need to rerun phase3 with proper anomaly detection before replay training begins. Add as preflight step.

**GT eligibility for replay comparison**:
```python
def gt_is_reference_eligible(phase3_entry):
    return (
        phase3_entry["converged"]
        and not phase3_entry["surface_changed"]
        and not phase3_entry["pre_dissociated"]
        and not phase3_entry["dissociated"]
        and not phase3_entry["desorbed"]
        and not phase3_entry["intercalated"]
    )
```
Systems whose (best) GT config fails this are **excluded from replay target comparison** — we have no trustworthy reference to beat.

## 26. Per-system GT aggregation (for replay success check)

The LMDB has one entry per (slab, ads, config) triple. Replay searches global minima per (slab, ads) system, so:
```python
E_gt_min[system_key] = min(
    phase3[idx]["E_sys"]
    for idx in entries_of(system_key)
    if gt_is_reference_eligible(phase3[idx])
)
```
Success: model's UMA-relaxed `E_sys` for that system beats `E_gt_min[system_key]` by ≥ δ. This aggregation is pre-computed once before training (`scripts/build_replay_gt_index.py`).

## 27b. Eval budget scale-up trigger

`ReplayScheduler` monitors each eval's `new_samples_count`:

```python
class ReplayScheduler:
    def should_scale_up(self, history: List[int]) -> bool:
        if len(history) >= 5:
            return True                       # (b) hard floor: 5 evals done
        if len(history) >= 3:
            last3 = history[-3:]
            mean = sum(last3) / 3
            if max(abs(x - mean) for x in last3) / max(mean, 1) <= 0.3:
                return True                    # (a) plateau: ±30% within 3 evals
        return False
```

Initial budget: **500 systems × 3 placements** (≈45 min/eval @ 4 GPUs).
After trigger: **2000 systems × 5 placements** (≈3-4 h/eval @ 4 GPUs).

## 27. Final open questions (need confirmation)

1. **δ default 0.1 eV** — 네가 UMA 논문에서 정확한 per-system MAE 값 (예: 0.22 eV) 확인해서 알려주면 그대로 박을게. 우선은 0.1 eV 로 구현 후 override.
2. **Eval budget 초기값** — 첫 몇 eval 은 **500 systems × 3 placements (≈45분/eval with 4 GPUs)** 으로 시작, stabilize 되면 `2000 × 5` 로 scale-up. 이 점진 확대 OK?
3. **Flag B 대응** — phase3 flag 가 제대로 계산된 건지 preflight check (100 샘플 재확인) 를 첫 단계로 포함. OK?
4. **OC20-Dense 병합 파일** — `data/processed/oc20dense.lmdb` 로 새로 만들지 (oc20dense_train + oc20dense_val concat), 아니면 두 파일 모두 읽는 래퍼 클래스로? (권장: concat 한 번에 새 lmdb 생성, cleaner)
5. **Replay 저장 경로** 재확인: `runs/<run_name>/replay_buffer.pkl` + `runs/<run_name>/replay_new_samples_ep{N}.pkl`. OK?
