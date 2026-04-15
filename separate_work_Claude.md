# AdsorbGen v2 마이그레이션 — 병렬 작업 분할 (Claude × Codex)

이 문서는 [modify_architecture.md](./modify_architecture.md)에 정리된 v2 아키텍처 마이그레이션 작업을 **Claude와 Codex가 동시에 진행할 때의 분할 규칙**을 기술한다.

구조적 배경(왜 v2로 가는가, 어떤 코드를 빼는가, DoD, regression test 목록 등)은 전부 `modify_architecture.md`에 있다. 이 파일은 그 계획을 **두 agent가 충돌 없이 병렬로 실행하는 방법**만 다룬다.

---

## 1. 분할 원칙

두 agent가 **같은 파일을 절대 동시에 수정하지 않게** 수직 분할한다. 분할 축은 "model core vs. train/inference 배선".

- **Claude (Agent A)** — 새 모델 구현 + factory. PyTorch `nn.Module` 작성 업무만 한다. 학습 루프, checkpoint, CLI에는 관심 없다.
- **Codex (Agent B)** — 기존 파이프라인에 v2를 배선 + 호환성 버그 수정 + regression test. Model v2 내부 구현은 보지 않는다.

이 축이 가능한 이유: v2의 `forward` 시그니처가 v1과 동일하게 유지되므로, Agent B는 v2를 "이름이 다른 같은 모델"로 취급할 수 있다.

---

## 2. 파일 소유권

| 파일 | 상태 | 소유자 | 비고 |
|---|---|---|---|
| `adsorbgen/model_v2.py` | 신규 | **Claude** | `DiTDenoiserV2Config` + `DiTDenoiserV2` |
| `adsorbgen/model_factory.py` | 신규 | **Claude** | `build_model(cfg)` 단일 함수 |
| `adsorbgen/tests/test_model_v2.py` | 신규 | **Claude** | v2 단위 테스트 |
| `adsorbgen/train.py` | 수정 | **Codex** | `--arch`, factory 호출, `args.json` 저장, resume arch 검사, safe_globals |
| `adsorbgen/inference.py` | 수정 | **Codex** | `_extract_state_dict`, arch 분기, v1 loader 버그 수정 |
| `adsorbgen/tests/test_checkpoint_compat.py` | 신규 | **Codex** | v1 backward-compat + resume 검사 regression tests |
| `adsorbgen/model.py` | — | **금지** | v1은 한 줄도 수정하지 않는다 |
| `adsorbgen/transformer.py` | — | **금지** | 재사용 대상, 수정 없음 |
| `adsorbgen/flow.py` | — | **금지** | 수정 없음 |
| `adsorbgen/dataset.py` | — | **금지** | 수정 없음 |
| `adsorbgen/eval.py` | — | **금지** | 수정 없음 |
| `adsorbgen/tests/test_flow_matching.py` | — | **금지** | 이번 PR에서는 건드리지 않는다. architecture-agnostic 리팩터는 follow-up PR로 연기 |
| `adsorbgen/tests/test_eval.py` | — | **금지** | 수정 없음 |
| `adsorbgen/__init__.py` | — | **금지** | 새 export 추가 금지 |
| `adsorbgen/scripts/**` | — | **금지** | CLI 확장으로 기존 스크립트는 자동 호환되어야 함 |

**머지 충돌 표면: 0건**. 두 agent가 규칙을 지키면 파일 단위로 어떤 겹침도 없다.

---

## 3. 인터페이스 계약 (lock)

아래 public API는 **병렬 작업 시작 전에 고정**되며, 병렬 기간 동안 **변경 금지**. 필드 추가가 필요하다고 판단되면 양쪽 agent를 정지시키고 사용자에게 에스컬레이션 후 재동기화.

### 3.1 `adsorbgen/model_v2.py` — Claude가 구현, Codex는 import만

```python
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn


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


class DiTDenoiserV2(nn.Module):
    def __init__(self, cfg: DiTDenoiserV2Config): ...

    def forward(
        self,
        pos: torch.Tensor,              # (B, N, 3)
        delta_t: torch.Tensor,          # (B, N, 3)
        t: torch.Tensor,                # (B,)
        atomic_numbers: torch.Tensor,   # (B, N) long
        tags: torch.Tensor,             # (B, N) long
        movable_mask: torch.Tensor,     # (B, N) bool
        pad_mask: torch.Tensor,         # (B, N) bool
        cell: torch.Tensor,             # (B, 3, 3)
        delta_e: Optional[torch.Tensor] = None,      # (B,)
        cond_drop: Optional[torch.Tensor] = None,    # (B,) bool
    ) -> torch.Tensor:                  # (B, N, 3)
        ...
```

약속:
- **forward 시그니처는 v1 `DiTDenoiser.forward`와 완전히 동일**. 키워드 인자 이름과 순서 모두.
- Non-movable / padding 위치의 출력은 0으로 mask되어야 한다 (v1과 동일).
- 입력에 NaN/Inf가 있으면 `RuntimeError`를 발생시킨다 (v1과 동일).
- Config 필드 이름은 위 13개 그대로. Codex의 `args.json` 직렬화/역직렬화가 이 이름에 의존한다.

### 3.2 `adsorbgen/model_factory.py` — Claude가 구현, Codex가 호출

```python
import torch.nn as nn

from adsorbgen.model import DiTDenoiser, DiTDenoiserConfig
from adsorbgen.model_v2 import DiTDenoiserV2, DiTDenoiserV2Config


def build_model(model_cfg) -> nn.Module:
    if isinstance(model_cfg, DiTDenoiserConfig):
        return DiTDenoiser(model_cfg)
    if isinstance(model_cfg, DiTDenoiserV2Config):
        return DiTDenoiserV2(model_cfg)
    raise TypeError(
        f"Unknown model config type: {type(model_cfg).__name__}"
    )
```

약속:
- 외부 노출 API는 `build_model` 단 하나. 이외 함수/클래스는 추가하지 않는다.
- Codex는 `train.py`의 `AdsorbGenModule.__init__` (현재 `line 64`)과 `inference.py`의 모델 instantiate 지점에서 `build_model(model_cfg)` 한 줄로만 사용한다.

---

## 4. 실행 순서 (skeleton-first)

Stub 파일을 repo에 커밋하는 방식은 **쓰지 않는다** (나중에 지우는 작업이 발생하므로). 대신 **Claude가 먼저 skeleton만 빠르게 push**하여 Codex가 실존 파일을 import할 수 있게 한다.

### Phase 0 — 계약 고정 (사용자 주도, 5분)
- 사용자는 §3의 인터페이스 계약을 확정해서 두 agent에게 동일한 내용으로 전달한다.
- 이 시점 이후 계약 변경은 금지. 변경이 필요하면 양쪽 모두 정지 후 재시작.

### Phase 1 — Claude skeleton push (Claude 단독, ~10분)
Claude는 아래 최소 내용만 담은 첫 커밋을 push한다:

1. `adsorbgen/model_v2.py`
   - `DiTDenoiserV2Config` — **완성**. 필드 전부 기본값 포함.
   - `DiTDenoiserV2` — 클래스 정의 + `__init__` + `forward` 시그니처. 내부는 placeholder:
     ```python
     def forward(self, pos, delta_t, t, atomic_numbers, tags,
                 movable_mask, pad_mask, cell,
                 delta_e=None, cond_drop=None):
         # TODO: real implementation (Phase 2)
         out = torch.zeros_like(delta_t)
         return out * movable_mask.unsqueeze(-1).to(out.dtype)
     ```
   - 입력 NaN 검사와 출력 masking은 placeholder에도 이미 들어가 있어야 Codex의 smoke test가 통과한다.

2. `adsorbgen/model_factory.py` — **완성** (§3.2 코드 그대로).

3. `adsorbgen/tests/test_model_v2.py` — 빈 파일 또는 `test_imports_ok()` 정도의 최소 placeholder.

이 skeleton은 그 자체로 import 가능하고, `build_model(DiTDenoiserV2Config())`가 돌고, 더미 forward가 0 tensor를 돌려준다.

### Phase 2 — 병렬 본 작업 (Claude + Codex 동시)

**Claude:**
- `model_v2.py`의 `forward`를 실제 구현으로 채운다 (입력 임베딩, pair enrichment, 단일 DiT stack, output head).
- `test_model_v2.py`에 5개 unit test 추가 (shape / mask / finite / backward / pair semantics).

**Codex:**
- Phase 1에서 push된 skeleton을 import로 끌어다 쓰면서 `train.py` / `inference.py`를 수정.
- `test_checkpoint_compat.py`에 regression test 4종 추가.
- Codex의 통합 테스트가 **placeholder 구현으로도 통과**해야 한다 (forward가 0 tensor라도 shape/mask/checkpoint 로드는 검증 가능).

### Phase 3 — 머지 순서 (사용자 주도)

1. **Claude의 완성 PR 머지 먼저.** Phase 1 skeleton이 Phase 2에서 실구현으로 채워진 최종 상태.
2. **Codex의 완성 PR 머지.** Claude PR 머지 commit을 base로 rebase되어 있어야 한다. Codex의 CI는 이 시점에 실구현과 맞물려 다시 한 번 돌아야 한다.
3. **최종 통합 검증** (사용자):
   - 전체 test suite 실행.
   - 기존 v1 checkpoint로 `inference.py` sampling이 통과하는지 수동 확인 (DoD #6).
   - `--arch v2`로 짧은 학습 실행, NaN 없이 첫 몇 step 통과 확인.

---

## 5. Claude 작업지시 (Agent A)

### Scope
`model_v2.py`, `model_factory.py`, `tests/test_model_v2.py` **세 파일만**. 그 외 어떤 파일도 수정 금지.

### 할 일
1. **Phase 1**: §4의 skeleton을 작성하고 commit / push.
2. **Phase 2**: `model_v2.py`의 `DiTDenoiserV2.forward`를 실제 구현으로 채운다. 구현 목표는 `modify_architecture.md`의 "v2 목표 구조" 섹션과 일치한다:
   - Token embedding (atom + tag + movable + pos_proj + xt_proj + cell_emb, pad mask 적용)
   - Pair feature build (MIC diff, dist, non_bulk mask, ads_pair) — `adsorbgen.model._pair_diff_mic` 재사용 가능
   - 1회 outer-product pair enrichment + LayerNorm
   - Conditioning: `TimestepEmbedder(t) + DeltaEEmbedder(delta_e) * (1 - cond_drop)`
   - 단일 `DiT` stack (from `adsorbgen.transformer`) — `dim`, `pair_dim`, `depth`, `num_heads` 파라미터만 사용
   - LayerNorm → `Linear(dim, 3)` zero-init → movable mask
3. **test_model_v2.py**에 5개 테스트 추가:
   - `test_shape`: `(B=2, N=32)` 더미 입력 → 출력 `(2, 32, 3)`
   - `test_mask_zeroed`: padding/non-movable 위치 출력이 정확히 0
   - `test_dtype_preserved`: float32 입력 → float32 출력
   - `test_gradient_flow`: `out.sum().backward()` 후 모든 학습 파라미터에 grad 존재
   - `test_numerics`: 입력 NaN → `RuntimeError`, 정상 입력 → 출력 finite

### 재사용 대상 (읽기 전용 import)
- `adsorbgen.transformer`: `DiT`, `DiTBlock`, `MaskedSelfAttention`, `TimestepEmbedder`
- `adsorbgen.model`: `DeltaEEmbedder`, `CellEmbedder`, `_pair_diff_mic`, `DiTDenoiserConfig` (factory용)

### 하지 말 것
- `model.py`, `transformer.py`, `flow.py`, `dataset.py`, `eval.py`, `train.py`, `inference.py`, `test_flow_matching.py`, `test_eval.py`, `__init__.py` — 그 어떤 수정도 금지.
- §3 계약 변경 금지 (필드 추가/삭제/rename 포함).
- `encoder`/`trunk`/`decoder`로 이름 짓지 말 것. v2의 본질은 단일 uniform stack임.
- v1 코드를 참고하는 것은 OK, 하지만 `atom_to_token`, `trunk_s_init`, `token_to_atom` 등의 보조 모듈은 **v2에 가져오지 말 것**.

### 완료 기준
- `pytest adsorbgen/tests/test_model_v2.py` 5개 전부 통과.
- `build_model(DiTDenoiserV2Config())` 가 실제 `DiTDenoiserV2` 인스턴스를 돌려준다.
- `modify_architecture.md`의 DoD 중 #1, #2, #3에 해당.

---

## 6. Codex 작업지시 (Agent B)

### Scope
`train.py`, `inference.py`, `tests/test_checkpoint_compat.py` **세 파일만**. 그 외 어떤 파일도 수정 금지.

### 할 일

#### 6.1 `inference.py`
1. **신규 함수 `_extract_state_dict(state)`** 추가: Lightning (`state["state_dict"]` + `"model."` prefix stripping), 구 custom (`state["model"]`), raw 세 포맷 dispatch. 현재 `inference.py:159-160`의 단순 로직을 대체. 이 변경만으로 **현재 존재하는 Lightning `last.ckpt` 로드 버그가 수정**된다.
2. **`_resolve_model_cfg` 확장** (`inference.py:45-69`):
   - `a.get("arch", "v1")`로 분기.
   - v1 branch: 기존 로직을 유지하되 **`delta_e_freq_dim`, `num_elements`, `num_tags` 복원 누락 버그 수정**. `DiTDenoiserConfig(**a)` 형태의 `**unpack` 패턴으로 재작성해서 필드 추가 시 자동 반영되게 한다 (activation_checkpointing 같은 runtime-only 필드는 필요 시 pop).
   - v2 branch: `a["model_config"]`를 `DiTDenoiserV2Config(**model_config)`로 unpack.
3. **모델 instantiate** (`inference.py:166`): `DiTDenoiser(model_cfg)` → `build_model(model_cfg)`로 교체 (`from adsorbgen.model_factory import build_model`).
4. **`add_safe_globals`** 호출 추가: `DiTDenoiserV2Config`를 포함. `train.py`와 동일 리스트.
5. (선택) **`--train-args-json` CLI 인자** 추가. `inference.py:68`의 주석이 이미 암시하는 override 경로.

#### 6.2 `train.py`
1. **Import 추가**: `from adsorbgen.model_v2 import DiTDenoiserV2, DiTDenoiserV2Config`, `from adsorbgen.model_factory import build_model`.
2. **`AdsorbGenModule.__init__`** (`train.py:64`): 하드코딩된 `self.model = DiTDenoiser(model_cfg)` → `self.model = build_model(model_cfg)`로 교체.
3. **CLI 인자 추가**:
   - `--arch {v1,v2}` (default: `v1`, 기존 스크립트 호환 유지)
   - `--dim`, `--pair-dim`, `--depth`, `--num-heads` (v2용)
4. **`build_config` 함수 확장**: `args.arch`를 보고 `DiTDenoiserConfig` 또는 `DiTDenoiserV2Config` 반환.
5. **`add_safe_globals`** (`train.py:317`): `DiTDenoiserV2Config` 추가.
   ```python
   torch.serialization.add_safe_globals([
       DiTDenoiserConfig,
       DiTDenoiserV2Config,
       FlowConfig,
   ])
   ```
6. **Resume arch 검사** (`train.py:340` auto-resume 분기 **직전**):
   - `out_dir/args.json`이 존재하면 먼저 읽어 기존 run의 `arch` 확인 (`a.get("arch", "v1")`).
   - 현재 `args.arch`와 불일치하면 `RuntimeError("arch mismatch: out_dir has arch={old}, but --arch={new} given. Use a different --out, or remove the directory.")`로 fail-fast.
   - 일치하면 resume 진행. 이 경우 `args.json`은 덮어쓰지 않는다 (이미 존재하는 파일이 source of truth).
7. **`args.json` 자동 저장**: `out_dir`이 신규이고 `args.json`이 없을 때, 학습 시작 직전에 저장한다.
   - v1: flat schema, `arch` 필드 **없음** (기존 파일 포맷과 구별 불가하게 유지).
   - v2: `{"arch": "v2", "model_config": asdict(model_cfg)}`.
   - 저장은 `asdict(model_cfg)` 기반으로 해서 config dataclass에 필드가 추가되면 자동 반영.

#### 6.3 `tests/test_checkpoint_compat.py` (신규)
regression test 4종:

1. **v1 raw state_dict 로드**: 더미 v1 `args.json` (arch 필드 없는 flat schema) + raw `state_dict` → `_resolve_model_cfg` → `build_model` → `load_state_dict` → 더미 forward 유한성 확인.
2. **v1 Lightning `last.ckpt` 로드 (★ 현재 코드 버그 regression)**: `{"state_dict": {"model.atom_embed.weight": ..., ...}, "epoch": 0, "global_step": 0}` 형태로 저장. `_extract_state_dict`가 올바른 key set을 반환하는지 + end-to-end 로드 + forward 통과 확인. 이 테스트는 **현재 main에서는 실패해야 하고**, Codex의 수정 후 통과해야 한다.
3. **비-default 필드 완전 복원**: `delta_e_freq_dim=128`, `num_elements=95` 같은 non-default 값으로 `args.json` 저장. 로드 후 `model.delta_e_embedder.frequency_embedding_dim == 128`, `model.atom_embed.num_embeddings == 95` 확인.
4. **Resume arch 검사**: 임시 `out_dir`에 v1 `args.json` + 더미 `last.ckpt` 생성. `train.py`의 resume 로직을 호출할 때 `args.arch == "v2"`면 `RuntimeError`로 fail-fast, `args.arch == "v1"`이면 정상 진입 확인.

### 하지 말 것
- `model.py`, `model_v2.py`, `model_factory.py`, `transformer.py`, `flow.py`, `dataset.py`, `eval.py`, `test_flow_matching.py`, `test_eval.py`, `__init__.py` — 그 어떤 수정도 금지.
- v2 모델 내부 구현을 추측해서 test에 박지 말 것 (shape/forward signature만 신뢰).
- `test_flow_matching.py`를 리팩터하지 말 것. 새 테스트는 전부 `test_checkpoint_compat.py`에 추가한다.
- `args.json` v1 schema의 필드 이름이나 기본값을 변경하지 말 것 (기존 checkpoint 호환성 파괴).
- v1 코드 경로를 제거하거나 deprecation warning을 추가하지 말 것.

### 완료 기준
- `pytest adsorbgen/tests/test_checkpoint_compat.py` 4개 전부 통과.
- 기존 v1 Lightning `last.ckpt`를 코드/파일 수정 없이 `inference.py`로 로드해서 sampling 통과.
- `python train.py --arch v2 --out /tmp/test_v2 ...` 로 짧은 학습 1 step이 NaN 없이 돈다 (Phase 3에서 Claude 실구현과 통합된 뒤).
- `modify_architecture.md`의 DoD 중 #4, #5, #6, #7에 해당.

---

## 7. 금지 / 연기 사항 정리

### 이번 PR에서 건드리지 않는 파일
`model.py`, `transformer.py`, `flow.py`, `dataset.py`, `eval.py`, `tests/test_flow_matching.py`, `tests/test_eval.py`, `adsorbgen/__init__.py`, `scripts/**`.

### Follow-up PR로 연기되는 작업
- `test_flow_matching.py`의 architecture-agnostic 리팩터 (v1 코드가 살아 있는 한 이 테스트는 그대로 통과하므로 시급하지 않다).
- v1 `DiTDenoiser` 및 그 보조 모듈 제거 (v2가 실전에서 검증되고 기존 checkpoint 재학습 경로가 확보된 후에 별도 PR).
- `modify_architecture.md` Step 6의 compressed v2 ablation (`384/128/14` 등).
- 후속 도메인 실험 (distance RBF, adsorbate-centric cross-attention, frozen atom KV-only 등).

---

## 8. 사용자 체크리스트

병렬 작업을 시작하기 전에 사용자가 확인해야 할 것:

- [ ] §3의 인터페이스 계약을 두 agent에게 동일한 내용으로 전달했는가?
- [ ] 두 agent가 서로 다른 브랜치에서 작업하는가? (Claude 브랜치, Codex 브랜치 분리)
- [ ] Claude가 Phase 1 skeleton을 push한 시점을 Codex의 브랜치 base로 사용하는가?
- [ ] Codex 브랜치가 Claude 브랜치의 최종 commit을 base로 rebase되어 있는가? (머지 직전)
- [ ] 머지 순서는 Claude PR → Codex PR 인가?
- [ ] `modify_architecture.md`의 DoD 전체가 통합 검증 단계에서 체크되는가?

이 여섯 개가 만족되면 병렬 작업 분할은 성공적으로 끝난다.
