# AdsorbGen GPT Split Plan

## Goal

Use two coding agents on the same AdsorbGen v2 migration with:

- zero shared write targets
- a fixed interface contract
- low merge risk
- clear ownership

This document is the final recommended split.

---

## Final Recommendation

Use a **two-lane split**:

- `Claude`: model-core lane
- `Codex`: train/inference integration lane

Do **not** let both agents edit the same file.

The safest execution order is:

1. Freeze the interface contract in this document.
2. `Claude` implements the new model files.
3. `Codex` starts from the commit that already contains those model files and wires v2 into training/inference/checkpoint compatibility.
4. Run final integration verification after both lanes land.

This is slightly less parallel than a stub-based workflow, but it is safer and simpler. For this task, avoiding stub drift is worth more than squeezing out a bit of overlap.

---

## File Ownership

### Claude owns

Files `Claude` may create or modify:

- `AdsorbGen/adsorbgen/model_v2.py`
- `AdsorbGen/adsorbgen/tests/test_model_v2.py`

Files `Claude` must not modify:

- `AdsorbGen/adsorbgen/train.py`
- `AdsorbGen/adsorbgen/inference.py`
- `AdsorbGen/adsorbgen/model.py`
- `AdsorbGen/adsorbgen/model_factory.py`
- `AdsorbGen/adsorbgen/transformer.py`
- `AdsorbGen/adsorbgen/tests/test_flow_matching.py`
- `AdsorbGen/adsorbgen/tests/test_checkpoint_compat.py`

### Codex owns

Files `Codex` may create or modify:

- `AdsorbGen/adsorbgen/train.py`
- `AdsorbGen/adsorbgen/inference.py`
- `AdsorbGen/adsorbgen/model_factory.py`
- `AdsorbGen/adsorbgen/tests/test_checkpoint_compat.py`

Files `Codex` must not modify:

- `AdsorbGen/adsorbgen/model.py`
- `AdsorbGen/adsorbgen/model_v2.py`
- `AdsorbGen/adsorbgen/transformer.py`
- `AdsorbGen/adsorbgen/tests/test_model_v2.py`
- `AdsorbGen/adsorbgen/tests/test_flow_matching.py`

### Everyone leaves alone

These are out of scope for this split:

- `AdsorbGen/adsorbgen/dataset.py`
- `AdsorbGen/adsorbgen/flow.py`
- `AdsorbGen/adsorbgen/eval.py`
- `AdsorbGen/adsorbgen/__init__.py`
- `ARCHITECTURE.txt`

---

## Interface Contract

This contract must not change during the split.

### Module path

`Claude` must export the new model from:

```python
from adsorbgen.model_v2 import DiTDenoiserV2, DiTDenoiserV2Config
```

### Config contract

`DiTDenoiserV2Config` must expose exactly these fields and defaults:

```python
from dataclasses import dataclass


@dataclass
class DiTDenoiserV2Config:
    dim: int = 512
    pair_dim: int = 128
    depth: int = 13
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    num_elements: int = 100
    num_tags: int = 3
    sigma: float = 0.5
    delta_e_max: float = 2.0
    delta_e_freq_dim: int = 256
    activation_checkpointing: bool = False
```

### Forward contract

`DiTDenoiserV2.forward(...)` must match v1 exactly:

```python
def forward(
    self,
    pos: torch.Tensor,
    delta_t: torch.Tensor,
    t: torch.Tensor,
    atomic_numbers: torch.Tensor,
    tags: torch.Tensor,
    movable_mask: torch.Tensor,
    pad_mask: torch.Tensor,
    cell: torch.Tensor,
    delta_e: Optional[torch.Tensor] = None,
    cond_drop: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ...
```

Expected output shape:

```python
(B, N, 3)
```

### Behavior contract

`DiTDenoiserV2` must preserve these behaviors:

- zero-init output head
- output zeroed on non-movable atoms
- NaN/Inf guard behavior consistent with v1
- same mask semantics as v1
- same conditioning argument semantics as v1

If either agent thinks this contract must change, stop and re-align before more code is written.

---

## Factory Contract

`Codex` owns the factory and should place it in:

- `AdsorbGen/adsorbgen/model_factory.py`

Recommended API:

```python
from adsorbgen.model import DiTDenoiser, DiTDenoiserConfig
from adsorbgen.model_v2 import DiTDenoiserV2, DiTDenoiserV2Config


def build_model(model_cfg):
    if isinstance(model_cfg, DiTDenoiserConfig):
        return DiTDenoiser(model_cfg)
    if isinstance(model_cfg, DiTDenoiserV2Config):
        return DiTDenoiserV2(model_cfg)
    raise TypeError(f"Unknown model config type: {type(model_cfg).__name__}")
```

Why `model_factory.py` instead of `models/__init__.py`:

- less naming confusion with existing `adsorbgen.model`
- no need to introduce a new subpackage
- simple ownership: one integration file, owned by `Codex`

---

## Claude Scope

`Claude` is responsible only for the new model implementation.

### Deliverables

- `adsorbgen/model_v2.py`
- `adsorbgen/tests/test_model_v2.py`

### What Claude should implement

- `DiTDenoiserV2Config`
- `DiTDenoiserV2`
- reuse of existing `transformer.py`
- uniform DiT architecture described in `modify_architecture.md`
- model-local tests only

### What Claude should not do

- no edits to training CLI
- no edits to inference loader
- no checkpoint compatibility work
- no `args.json` logic
- no factory integration in train/inference

### Claude DoD

- file imports cleanly
- config matches contract exactly
- forward signature matches contract exactly
- output shape is correct
- non-movable outputs are zeroed
- forward/backward are finite on dummy inputs
- tests in `test_model_v2.py` pass

---

## Codex Scope

`Codex` is responsible for everything that makes v2 usable without breaking v1.

### Deliverables

- `adsorbgen/train.py`
- `adsorbgen/inference.py`
- `adsorbgen/model_factory.py`
- `adsorbgen/tests/test_checkpoint_compat.py`

### What Codex should implement

- `--arch {v1,v2}` in training
- config selection for v1 vs v2
- `args.json` autosave
- resume arch mismatch guard
- Lightning `last.ckpt` state dispatch
- v1 metadata restore bug fixes
- arch-aware model construction through `build_model(...)`
- safe globals updates for Lightning serialization

### Required checklist for Codex

- replace hardcoded `DiTDenoiser(model_cfg)` in `AdsorbGenModule` with `build_model(model_cfg)`
- add `DiTDenoiserV2Config` to `torch.serialization.add_safe_globals(...)`
- if needed, add matching safe-global handling on inference load paths too
- implement `_extract_state_dict(...)` in inference for:
  - Lightning `state["state_dict"]`
  - legacy `state["model"]`
  - raw state dict
- fix v1 loader reconstruction for:
  - `delta_e_freq_dim`
  - `num_elements`
  - `num_tags`
- write `args.json` automatically for new runs
- block resume when existing run arch and requested `--arch` disagree

### What Codex should not do

- no edits inside `model_v2.py`
- no edits inside `test_model_v2.py`
- no edits to `transformer.py`
- no edits to `test_flow_matching.py` in this phase

### Codex DoD

- v1 still trains/loads/samples
- v2 can be selected from train/inference
- Lightning `last.ckpt` loads in inference
- legacy custom checkpoint format still loads
- raw state dict format still loads
- arch mismatch on resume fails before overwriting metadata
- compatibility tests pass

---

## Test Split

### Claude test file

- `adsorbgen/tests/test_model_v2.py`

Use it for:

- v2 shape tests
- mask behavior
- finite forward
- finite backward
- output zero-init behavior

### Codex test file

- `adsorbgen/tests/test_checkpoint_compat.py`

Use it for:

- raw v1 state dict load
- Lightning `last.ckpt` load
- legacy `state["model"]` load
- non-default config field restore
- resume arch mismatch fail-fast

### Do not touch in this split

- `adsorbgen/tests/test_flow_matching.py`

Reason:

- v1 remains intact
- changing this file adds merge/review risk
- test cleanup can be a later follow-up PR

---

## Merge Order

Recommended merge order:

1. `Claude` PR lands first
   - adds `model_v2.py`
   - adds `test_model_v2.py`
   - should not affect existing behavior
2. `Codex` rebases on that commit and lands second
   - adds factory
   - wires `train.py` and `inference.py`
   - adds checkpoint compatibility tests
3. run final integration verification

Why this order:

- no stub files needed
- no drift between temporary API and real API
- integration lane can import the actual model implementation

---

## Final Verification

After both lanes are merged, run:

1. Existing v1 tests that matter for compatibility
2. New `test_model_v2.py`
3. New `test_checkpoint_compat.py`
4. One v1 inference smoke using an existing-style checkpoint
5. One short v2 training smoke

Success means:

- v1 behavior still works
- v2 can be instantiated and trained
- checkpoint loading works across all supported formats
- no file ownership conflicts occurred during the split

---

## Summary

This is the final recommended workflow:

- `Claude` builds the new model only
- `Codex` owns integration, compatibility, and factories
- no shared write targets
- no stub workflow
- merge `Claude` first, then `Codex`

If this split is followed strictly, merge conflicts should be near zero and the migration risk stays concentrated in one place at a time.
