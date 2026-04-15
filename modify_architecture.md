# AdsorbGen 아키텍처 수정 권고안

## 핵심 결론

현재 `DiTDenoiser`의 **encoder-trunk-decoder 3-stage 구조**는
AdsorbGen에서는 유지 이점보다 유지비가 더 크다. 따라서 **단일 균일 DiT backbone**
(`uniform DiT`)으로 단순화하는 방향이 맞다.

다만 첫 마이그레이션은 다음 원칙을 반드시 지켜야 한다.

1. **구조 단순화와 capacity 변경을 분리한다.**
2. **v1을 즉시 교체하지 말고 v2를 병행 도입한다.**
3. **config / checkpoint / inference 계약을 같이 versioning한다.**

이 문서는 위 3가지를 만족하는 **실행 가능한 v2 마이그레이션 계획**이다.

---

## 왜 바꾸는가

현재 구조는 [AdsorbGen/adsorbgen/model.py](AdsorbGen/adsorbgen/model.py)의
AtomMOF식 3-stage 설계를 거의 그대로 가져온 것이다.

하지만 AdsorbGen의 현재 구현에서는:

- atom과 token이 사실상 **1:1**이다.
- `atom_to_token`, `token_to_atom`은 aggregation이 아니라 projection에 가깝다.
- trunk init의 outer-product enrichment도 계층적 tokenization이 아니라
  단일 dense batch 위의 추가 변환으로 작동한다.

즉 현재 구조는 실질적으로:

- `atom_s=256` 공간에서 얕은 DiT
- `token_s=512` 공간으로 projection
- 큰 DiT
- 다시 `atom_s=256`으로 projection
- pair 없는 얕은 DiT

를 순서대로 붙인 형태다.

이 구조는 실험을 느리게 만든다.

- 바꿔야 할 모듈이 많다.
- config가 stage별로 분산돼 있다.
- 테스트도 stage 이름에 암묵적으로 묶여 있다.
- 새 실험을 넣을 때 구조 변경과 도메인 변경이 같이 일어난다.

반면 AdsorbGen에서 반드시 보존해야 할 도메인 요소는 3-stage가 아니다.

- AttentionPairBias
- MIC pair geometry
- adaLN-Zero conditioning
- `pad_mask` / `movable_mask`
- `delta_e` conditioning + CFG drop
- zero-init output head

따라서 **도메인 요소는 유지하고, stage hierarchy만 제거하는 것**이 가장 합리적이다.

---

## 현재 계획의 문제점

기존 초안의 방향 자체는 맞지만, 그대로 가면 해석이 어려워진다.

### 1. 구조 단순화와 모델 축소가 섞여 있다

초안의 기본 제안은 대략:

- `dim=384`
- `pair_dim=128`
- `depth=14`

인데, 이 조합은 현재 기본 모델보다 **상당히 작은 모델**이다.

실측 기준:

- 현재 기본 v1 파라미터 수: 약 **66.5M**
- `384 / 128 / 14` 단일 DiT 추정 파라미터 수: 약 **40.0M**

즉 이 변경은 “구조 개선” 실험이 아니라 사실상 **구조 변경 + 용량 축소 실험**이 된다.
성능 차이가 나도 원인을 분리하기 어렵다.

### 2. 모델 밖 계약 변경 비용이 생각보다 크다

아래 파일들이 현재 3-stage config에 직접 의존한다.

- `adsorbgen/train.py`
- `adsorbgen/inference.py`
- `adsorbgen/tests/test_flow_matching.py`

특히 `inference.py`는 checkpoint 옆 `args.json` 기반 복원을 기대하는데,
현재 `train.py`는 그 파일을 명시적으로 저장하지 않는다.

따라서 이번 변경은 `model.py`만의 문제가 아니라:

- config schema
- checkpoint metadata
- inference reconstruction
- test contracts

까지 함께 정리해야 깔끔하다.

---

## 권장 목표 형태

### 아키텍처 원칙

첫 번째 v2는 아래 조건을 만족해야 한다.

1. **single-stream token representation**
2. **single shared pair representation**
3. **uniform DiT depth**
4. **forward signature 완전 유지**
5. **현재와 유사한 capacity 유지**

### 권장 backbone

첫 버전의 기본 후보는 아래가 가장 좋다.

```python
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

이 조합의 장점:

- 현재 기본 모델(`~66.5M`)과 거의 같은 크기(`~66.0M`)다.
- `pair_dim=128`을 유지해 pair 메모리를 키우지 않는다.
- 구조만 바뀌고 용량은 거의 유지되므로 비교가 공정하다.

**Pair stream 용량에 대한 주석 (대체로 안심해도 되는 이유):**

`pair_dim`이 v1의 `token_z=256`에서 v2의 `128`로 줄어드는 것이 걱정될 수 있지만, 파라미터 실측으로 보면 영향은 거의 없다.

v1 default(~66.5M) 구성 비중:

| 구성 | 파라미터 | 비중 |
|---|---|---|
| DiTBlock 내부 single-stream (Q/K/V, proj_o/g, MLP, adaLN) × 16 layers | ~65.0M | ~98% |
| 임베딩 + atom↔token projection + trunk init 등 | ~1.7M | ~2.5% |
| **Pair stream 관련 전부** (pair feat emb + atom_to_token_pair + trunk_z_init×2 + trunk_z_mlp + 매 layer `pair_norm`+`pair_to_heads`) | **~0.4M** | **~0.6%** |

이유는 `transformer.py:122-124`에 있다 — pair는 **transformed stream이 아니라 attention bias**로만 주입되고, layer마다 단 한 번 `Linear(Z → num_heads)`로 소비된다. 즉 `pair_to_heads(Z=256, H=8)`도 고작 2048 파라미터에 불과하다. Pair의 "무게"는 **runtime tensor `(B, N, N, Z)`의 메모리**지 파라미터 예산이 아니다.

따라서:

- `pair_dim` 256 → 128 변경으로 빠지는 파라미터는 ~250K 수준 (총 66M의 0.4%). Capacity 매칭 관점에서 **사실상 noise**다.
- 남는 이론적 쟁점은 "각 pair 엔트리의 representational bandwidth가 256-d에서 128-d로 줄었다"는 것뿐인데, 어차피 H=8 scalar bias로 사영되므로 128-d → 8-d projection에도 충분히 여유가 있다.
- **결론: `pair_dim=128`로 가도 거의 확실히 안전.** Step 5.5는 예방적 안전망이지 예상되는 경로가 아니다.

### 왜 `384 / 128 / 14`가 1차 기본값으로는 아쉬운가

그 설정은 나쁜 구조가 아니라, **1차 migration baseline으로는 너무 작은 모델**이다.

추천 순서는:

1. `512 / 128 / 13`으로 **capacity-matched v2**를 만든다.
2. v1과 성능 비교를 한다.
3. 그 다음에 `384 / 128 / 14` 같은 **compressed v2**를 별도 ablation으로 본다.

이 순서여야 실험 해석이 명확하다.

---

## v2 목표 구조

```text
[1] Token embedding
    tokens =
        atom_embed
      + tag_embed
      + movable_embed
      + pos_proj(pos)
      + xt_proj(pos + delta_t)
      + cell_embed(cell)
    then apply pad mask

[2] Pair feature build
    pair =
        emb_pair_pos(diff_mic)
      + emb_pair_dist(1 / (1 + d^2))
      + emb_pair_mask(non_bulk_mask)
      + emb_pair_ads(ads_pair)

[3] One-time pair enrichment
    pair = pair
         + pair_row_proj(tokens).unsqueeze(2)
         + pair_col_proj(tokens).unsqueeze(1)
    pair = LayerNorm(pair)

[4] Conditioning
    c = t_embedder(t) + delta_e_embedder(delta_e) * (1 - cond_drop)

[5] Uniform DiT
    x = DiT(tokens, c, pad_mask, pair)

[6] Output
    x = LayerNorm(x)
    out = Linear(dim -> 3)   # zero-init
    out = out * movable_mask
```

### 유지되는 것

- `transformer.py` 내부의 `DiT`, `DiTBlock`, `MaskedSelfAttention`
- AttentionPairBias
- MIC 기반 pair geometry
- timestep / delta_e / cell embedding
- adaLN-Zero
- mask semantics
- 현재 `forward(...)` 시그니처

### 1차 단계에서 하지 않는 것

- `pair_off_last_k`
- cross-attention
- RBF distance encoding
- dynamic pair recomputation from `x_t`
- frozen atom KV-only attention

이들은 **v2 baseline이 안정화된 다음** separate ablation으로 보는 게 맞다.

---

## 가장 좋은 마이그레이션 전략

### 권장 구현 전략: 병행 도입 + 버전 관리

기존 초안처럼 `model.py`를 바로 갈아끼우는 것보다 아래 방식이 낫다.

#### Phase 1. v2 병행 추가

- `adsorbgen/model_v2.py`에 `DiTDenoiserV2`, `DiTDenoiserV2Config` 추가
- 기존 `adsorbgen/model.py`는 그대로 유지

#### Phase 2. 모델 factory 도입

예시:

```python
def build_model_from_config(cfg):
    if cfg.arch == "v1":
        return DiTDenoiser(cfg)
    if cfg.arch == "v2":
        return DiTDenoiserV2(cfg)
    raise ValueError(...)
```

또는 `adsorbgen/models/__init__.py` 형태의 factory 모듈을 둔다.

핵심은 **호출부가 architecture를 선택 가능해야 한다**는 점이다.

#### Phase 3. config / checkpoint metadata versioning

필수 필드:

```python
arch: Literal["v1", "v2"]
```

추가 원칙:

- v1 config와 v2 config는 **동일 dataclass에 억지로 섞지 않는 것**이 더 낫다.
- 다만 `train.py`와 `inference.py`에서는 공통 JSON schema로 serialize 가능해야 한다.

v1 `args.json` (현재 `inference.py:45-69`가 읽는 형태 — **스키마 유지, 절대 변경 금지**):

```json
{
  "atom_s": 256, "atom_z": 128,
  "token_s": 512, "token_z": 256,
  "enc_depth": 2, "trunk_depth": 12, "dec_depth": 2,
  "enc_heads": 4, "trunk_heads": 8, "dec_heads": 4,
  "mlp_ratio": 4.0, "dropout": 0.0,
  "num_elements": 100, "num_tags": 3,
  "sigma": 0.5,
  "delta_e_max": 2.0, "delta_e_freq_dim": 256,
  "activation_checkpointing": false
}
```

**주의 — v1 metadata 복원 누락 수정 (기존 코드 버그):**

현재 `inference.py:50-66`은 `delta_e_freq_dim`, `num_elements`, `num_tags`를 **복원하지 않는다**. 지금까지는 default 값(`256/100/3`)이 맞아 떨어져서 돌아갔지만, 비-default 실험 checkpoint는 정확히 재구성되지 않을 수 있다. 이번 PR에서 **v1 loader에도 이 세 필드 복원 로직을 추가**한다. 기존 v1 `args.json`에 이 필드들이 없더라도 default를 쓰면 되므로 backward-compat을 깨지 않는다.

v2 `args.json` (신규):

```json
{
  "arch": "v2",
  "model_config": {
    "dim": 512,
    "pair_dim": 128,
    "depth": 13,
    "num_heads": 8,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "num_elements": 100,
    "num_tags": 3,
    "sigma": 0.5,
    "delta_e_max": 2.0,
    "delta_e_freq_dim": 256,
    "activation_checkpointing": false
  }
}
```

**Metadata 완전성 규칙 (엄격):**

**Config dataclass의 생성자에 들어가는 shape/behavior 필드는 전부 저장되어야 한다.** 저장 누락 필드가 있으면 비-default 학습 checkpoint가 나중에 재구성 불가능해진다. 구체적으로:

- `train.py`가 `args.json`을 쓸 때는 `asdict(model_cfg)` 결과를 그대로 dump하는 식으로 구현해서, config dataclass에 필드가 추가되면 자동으로 저장에도 반영되게 한다.
- `inference.py`의 loader는 `a.get(field, default)` 형태 대신 **dataclass의 default와 `**a` unpack**을 써서 로드한다. 이러면 "여기 한 필드 복원 빠졌네" 버그가 재발하지 않는다.
- 예외: runtime-only 필드(`activation_checkpointing` 같은 것)는 inference에서 의미 없으므로 로드 시 무시하거나 False로 강제해도 된다.

**Backward compatibility 규칙 (엄격):**

1. **`arch` 필드가 없으면 v1로 간주한다.** 기존에 수동으로 작성되어 돌아다니는 모든 v1 `args.json`은 `arch`가 없으므로, 이 규칙 하나로 **수정 없이 계속 로드되어야 한다**.
2. v1 `args.json`의 **필드 이름과 default 값은 건드리지 않는다.** `atom_s, atom_z, token_s, token_z, enc_depth, trunk_depth, dec_depth, enc_heads, trunk_heads, dec_heads, mlp_ratio, dropout, sigma, delta_e_max, activation_checkpointing` 전부 그대로 유지.
3. v2 metadata는 **`model_config` 중첩 dict 안에 넣는다.** v1은 flat이지만 v2는 중첩 — 구조만 보고도 어느 버전인지 즉시 판별 가능하고, v1 필드와 이름 충돌이 없다.
4. **v1 checkpoint는 v2 코드 릴리스 이후에도 영구 호환 대상이다.** v1을 지우거나 rename하지 않고, `adsorbgen/model.py`에 그대로 둔다.

#### Phase 4. `args.json`을 실제로 저장

현재 `inference.py`는 `args.json`이 있다고 가정하지만 `train.py`는 저장하지 않는다 — 지금까지는 사용자가 수동으로 옆에 붙여왔다는 뜻이다.

이번 작업에서는 `train.py`가 checkpoint 저장 디렉토리에 `args.json`을 **반드시 자동 기록**하도록 정리한다.

- v1 학습: 기존 flat schema로 저장 (`arch` 필드 없음 — 기존 파일 포맷과 구별 불가하게 유지).
- v2 학습: 위 중첩 schema로 저장 (`arch: "v2"` 포함).

이건 부수 작업이 아니라 **필수 마이그레이션 작업**이다.

#### Phase 5. Checkpoint 포맷 호환성 (현재 코드 버그 포함)

**현재 상태 (버그):**

`train.py`는 Lightning `ModelCheckpoint`를 쓰고 (`train.py:362-369`), 기본 산출물은 `out_dir/last.ckpt` 이다. 이 파일의 구조는 Lightning 표준:

```python
state = {"state_dict": {"model.atom_embed.weight": ..., ...}, "epoch": ..., "global_step": ..., ...}
```

그리고 `train.py:346-355`의 finetune 로드 경로는 세 가지 포맷을 전부 handling한다:

```python
if "state_dict" in state:       sd = {k.removeprefix("model."): v for k,v in state["state_dict"].items() if k.startswith("model.")}
elif "model" in state:          sd = state["model"]       # 구 custom 포맷
else:                           sd = state                # raw state_dict
```

**반면 `inference.py:159-160`은 오직 `state["model"]` 또는 raw만 처리한다:**

```python
sd = state["model"] if isinstance(state, dict) and "model" in state else state
```

즉 **유저가 학습 직후 생성된 `last.ckpt`를 `inference.py`에 그대로 넘기면 state_dict는 `{"epoch": ..., "global_step": ...}` 같은 엉뚱한 객체로 읽히고, 이후 `load_state_dict`가 unexpected keys 투성이로 실패한다.** 이건 이번 PR 이전부터 있던 버그지만, 호환성 계약을 정리하는 이번 PR에서 같이 수정해야 한다.

**결정: `inference.py`가 Lightning `last.ckpt`를 직접 로드할 수 있어야 한다.**

별도 export 단계를 도입하는 대안도 있지만, 그러면 기존 워크플로우(학습 → 바로 `inference.py --ckpt last.ckpt`)가 전부 깨진다. `inference.py`에 포맷 dispatch 로직 한 블록을 추가하는 것이 훨씬 작은 변경이다.

구체 스펙:

```python
# inference.py 에 추가
def _extract_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        # Lightning format: keys prefixed with "model."
        return {k.removeprefix("model."): v
                for k, v in state["state_dict"].items()
                if k.startswith("model.")}
    if isinstance(state, dict) and "model" in state:
        # Legacy custom format
        return state["model"]
    return state  # raw
```

그리고 `_resolve_model_cfg`의 `args.json` 탐색 경로도 Lightning `.ckpt`와 함께 쓸 수 있게 확장한다. 일반적으로 `out_dir/last.ckpt`와 `out_dir/args.json`이 **같은 디렉토리**에 있으므로 현재 로직(`ckpt_path.parent / "args.json"`)은 그대로 작동한다. 단, `out_dir` 바깥의 checkpoint를 명시적으로 넘길 때를 위해 `--train-args-json` 같은 override 인자를 하나 두어도 좋다 (`inference.py:68`의 주석이 이미 이 존재를 암시하고 있으므로 실제로 구현만 하면 된다).

#### Phase 6. Resume 시 arch 일치성 검사

**현재 상태:** `train.py:340`은 `out_dir/last.ckpt`가 있으면 **arch를 검사하지 않고 무조건 auto-resume** 한다. 이 상태에서:

- 유저가 `out_dir=runs/foo`에 v1 학습을 해두고,
- 나중에 `python train.py --out runs/foo --arch v2 ...`를 실행하면,

`last.ckpt`가 존재하므로 Lightning resume이 시작되는데, module은 v2로 만들어져 있으므로 **state_dict 키가 전부 맞지 않아 조용히 실패하거나, 최악의 경우 같은 `out_dir`의 `args.json`을 v2 스키마로 덮어써서 기존 v1 run의 metadata를 날린다.**

**규칙 (필수):**

1. `train.py`는 auto-resume 분기(`train.py:340`) 진입 **전에** `out_dir/args.json`이 존재하면 그것을 먼저 읽어 기존 run의 `arch`를 확인한다.
2. 현재 CLI의 `--arch`와 불일치하면 **fail-fast**: `RuntimeError("arch mismatch: out_dir has arch=v1, but --arch=v2 given. Use a different --out, or remove the directory.")`.
3. 기존 run에 `args.json`이 없으면(=`arch` 필드 없음) v1으로 간주하고 현재 `--arch`가 v1이 아니면 같은 방식으로 에러.
4. 일치할 때만 resume 진행. 이 경우 `args.json`은 덮어쓰지 않는다 (이미 존재하는 파일이 source of truth).
5. `out_dir`이 비어 있는 경우(신규 학습)에는 `args.json`을 새로 쓴다.

이 규칙은 Lightning의 optimizer/epoch/step resume 기능은 그대로 유지한 채, **잘못된 arch 조합만 차단**하는 최소 변경이다.

---

## 코드 단위 수정 포인트

### `adsorbgen/model.py`

현재 제거 후보였던:

- `atom_to_token`
- `atom_to_token_pair`
- `cond_proj`
- `trunk_s_init`
- `trunk_z_init_1`
- `trunk_z_init_2`
- `trunk_s_mlp`
- `trunk_z_mlp`
- `token_to_atom`
- `encoder / trunk / decoder` 분리

는 v2 안에서는 맞게 사라진다.

다만 **v1 코드에서는 당장 지우지 않는다**.

### `adsorbgen/train.py`

필수 변경:

- `--arch {v1,v2}` 추가
- v2용 config builder 추가
- 실험 metadata 저장 (`args.json` 또는 `model_config.json`)
- **`AdsorbGenModule`이 arch-aware 해야 한다** (아래 상세)
- **Lightning hyperparameter 직렬화에 v2 config 등록** (아래 상세)

권장 방향:

- v1 CLI 인자는 유지
- v2용 인자는 별도로 추가
  - `--dim`
  - `--pair-dim`
  - `--depth`
  - `--num-heads`

즉 "기존 실험은 안 깨고, 새 실험만 확장"하는 구조가 좋다.

**`AdsorbGenModule` arch-aware factory (현재 하드코딩 제거):**

현재 `train.py:49-65`의 `AdsorbGenModule.__init__`은 생성자 본문에서 `self.model = DiTDenoiser(model_cfg)`로 **v1을 하드코딩**하고 있다. 이 상태에서는 아무리 바깥에서 `--arch v2`를 받아도 모듈은 여전히 v1만 만든다.

수정: `AdsorbGenModule`이 `model_cfg`의 타입(또는 명시적 `arch` 인자)을 보고 올바른 모델 클래스를 instantiate한다.

```python
# adsorbgen/models/__init__.py
def build_model(model_cfg):
    if isinstance(model_cfg, DiTDenoiserConfig):
        return DiTDenoiser(model_cfg)
    if isinstance(model_cfg, DiTDenoiserV2Config):
        return DiTDenoiserV2(model_cfg)
    raise TypeError(f"Unknown model config type: {type(model_cfg).__name__}")

# train.py
class AdsorbGenModule(L.LightningModule):
    def __init__(self, model_cfg, flow_cfg, ...):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(model_cfg)   # ← factory 사용
        ...
```

이렇게 하면 `AdsorbGenModule`은 arch에 대해 중립이고, `build_model`만 v2 추가 시점에 한 번 확장하면 된다. `inference.py`도 같은 factory를 재사용한다.

**Lightning hyperparameter 직렬화 (torch safe_globals):**

현재 `train.py:63`의 `self.save_hyperparameters()`는 생성자에 넘긴 `model_cfg`를 그대로 Lightning hparams에 직렬화한다. 그리고 `train.py:317`에서 `torch.serialization.add_safe_globals([DiTDenoiserConfig, FlowConfig])`로 역직렬화 허용 클래스를 등록한다.

v2를 추가하면 **v2 config도 반드시 safe_globals에 등록해야 한다**. 빠뜨리면:

- v2 학습은 돌아가지만,
- 그 결과로 생긴 `last.ckpt`를 `torch.load`할 때 (resume/finetune/inference 모두) `DiTDenoiserV2Config`가 unknown class로 걸려 `UnpicklingError` 발생.

수정:

```python
# train.py
from adsorbgen.model import DiTDenoiser, DiTDenoiserConfig
from adsorbgen.model_v2 import DiTDenoiserV2, DiTDenoiserV2Config  # ← 새 import

torch.serialization.add_safe_globals([
    DiTDenoiserConfig,
    DiTDenoiserV2Config,   # ← 추가
    FlowConfig,
])
```

`inference.py`에서도 같은 등록이 필요하다 (현재 `inference.py`에는 없지만, Lightning `last.ckpt`를 직접 로드하게 되면 같은 이유로 필요해진다). Phase 5의 `_extract_state_dict` 수정과 세트로 같이 들어가야 한다.

### `adsorbgen/inference.py`

필수 변경:

- `_resolve_model_cfg`를 분기형으로 확장:
  ```python
  def _resolve_model_cfg(ckpt_path):
      a = json.load(open(ckpt_path.parent / "args.json"))
      arch = a.get("arch", "v1")  # ★ 없으면 무조건 v1
      if arch == "v1":
          return "v1", DiTDenoiserConfig(**{...기존 필드들...})
      elif arch == "v2":
          mc = a["model_config"]
          return "v2", DiTDenoiserV2Config(**mc)
      else:
          raise ValueError(f"Unknown arch: {arch}")
  ```
- 모델 instantiate 지점에서 `arch`를 읽고 `DiTDenoiser` vs `DiTDenoiserV2`를 분기 생성.
- `state_dict` 로드 시 `strict=False`는 지금처럼 유지 (`inference.py:167`).

**절대 깨지면 안 되는 것:**

- **기존에 작성되어 있는 v1 `args.json` (arch 필드 없는 flat schema)을 가진 v1 checkpoint**가 이번 PR 후에도 **zero-change**로 로드 / 샘플링 가능해야 한다.
- 이건 Best-effort가 아니라 **regression test**로 보호해야 한다 (아래 DoD 참조).

### `adsorbgen/tests/test_flow_matching.py`

이 파일은 현재 일부 테스트가 stage 구조를 암묵적으로 가정한다.

따라서 테스트를 두 층으로 나누는 것이 좋다.

1. **architecture-agnostic invariant tests**
   - output masking
   - zero-init output
   - finite forward
   - gradient flow
   - pair feature semantics

2. **architecture-specific smoke tests**
   - v1 instantiate / forward
   - v2 instantiate / forward

이렇게 해야 이후 구조를 바꿔도 핵심 테스트가 오래 살아남는다.

---

## 권장 실행 순서

### Step 1. 문서/메타데이터 계약 먼저 정리

- `v1`, `v2` naming 확정
- config schema 확정
- checkpoint metadata schema 확정

### Step 2. `DiTDenoiserV2` 추가

조건:

- `forward` 시그니처는 v1과 완전히 동일
- NaN / Inf guard 유지
- output masking 유지

### Step 3. `train.py` / `inference.py`를 arch-aware로 수정

반드시 포함:

- `--arch`
- config serialization
- checkpoint restore 분기

### Step 4. 테스트를 architecture-agnostic하게 정리

최소 추가 검증:

- v2 shape
- mask zeroing
- finite forward
- backward 통과
- pair feature shape / semantics

**v1 backward-compat regression test (필수, 3종):**

이 테스트 묶음이 떨어지면 기존 v1 checkpoint들이 깨졌다는 뜻이므로 전부 **merge blocker**.

1. **Raw state_dict 로드 경로**
   - 더미 v1 `args.json` (arch 필드 없는 flat schema) + raw `state_dict` torch.save로 저장.
   - `_resolve_model_cfg`가 `arch="v1"`로 resolve하는지 확인.
   - `DiTDenoiser` instantiate + state_dict 로드 + 더미 forward 유한성 확인.

2. **Lightning `last.ckpt` 포맷 로드 경로 (★ 현재 코드 버그 regression)**
   - 실제 `train.py`의 `ModelCheckpoint`가 만드는 구조를 흉내내서, `{"state_dict": {"model.atom_embed.weight": ..., ...}, "epoch": 0, "global_step": 0}` 형태로 저장.
   - `inference.py`의 새 `_extract_state_dict`가 `"model."` prefix를 벗기고 올바른 key set을 반환하는지 확인.
   - 전체 end-to-end 로드 + forward 통과.
   - 이 테스트는 **현재 코드에서는 반드시 실패해야 하고**, Phase 5 수정 후 통과해야 한다.

3. **비-default 필드 완전 복원**
   - `delta_e_freq_dim=128`, `num_elements=95` 같은 **non-default** 값으로 `args.json`을 만들고 저장.
   - 로드 후 `model.delta_e_embedder.frequency_embedding_dim == 128`, `model.atom_embed.num_embeddings == 95`인지 확인.
   - 현재 loader에서는 실패할 가능성이 높으며 (metadata 복원 누락 버그), Phase 3의 규칙 적용 후 통과해야 한다.

**Resume arch 검사 테스트 (Phase 6 regression):**

- 임시 `out_dir`에 v1 `args.json`과 더미 `last.ckpt`를 만든다.
- `train.py`의 resume 로직을 호출할 때 `--arch v2`를 주면 `RuntimeError("arch mismatch: ...")`로 fail-fast하는지 확인.
- `--arch v1`(또는 생략)일 때는 정상 resume 경로로 진입하는지 확인.

### Step 5. capacity-matched 비교

첫 비교는 아래 기준으로 한다.

- v1 default
- v2 `dim=512, pair_dim=128, depth=13, num_heads=8`

비교 항목:

- 파라미터 수
- forward / backward 안정성
- 짧은 학습 loss
- sample eval 메트릭

### Step 5.5. (Contingency, 실행 가능성 낮음) pair-bandwidth ablation

**이 Step은 실행될 가능성이 낮다.** 위 "Pair stream 용량에 대한 주석"에서 본 것처럼, pair stream은 전체 파라미터의 ~0.6%에 불과하고 `pair_dim`이 절반이 되어도 capacity 관점에서는 거의 noise다. 따라서 **기본 가정은 Step 5에서 바로 Step 6으로 넘어가는 것**이고, 이 단계는 "혹시 모를 경우"에 대비한 예방적 안전망일 뿐이다.

발동 조건: Step 5에서 v2가 v1에 비해 **애매하게 뒤쳐지고**, **그 원인이 불분명할 때에만** 실행한다. v1과 동등하거나 더 좋으면 건너뛴다 (이 경우가 훨씬 더 가능성 높음).

**명명 주의:** 이 ablation의 진짜 목적은 파라미터 수 매칭이 아니라 **pair feature의 representational bandwidth**(엔트리당 몇 차원 벡터인가) 확인이다. 파라미터 수로 보면 `pair_dim` 128 ↔ 256 차이는 ~250K에 불과해서 "pair-matched"라는 표현은 오해의 소지가 있다. 정확히는 pair bandwidth가 v1 trunk와 동일한 수준(256-d per entry)인 v2 variant를 돌려보는 것이다.

목적: 만에 하나 gap이 났을 때, 원인이 **구조 단순화**인지 **pair entry bandwidth 축소**인지 분리.

비교 설정 — **high-bandwidth v2 variant**:

```python
# pair-bandwidth ablation variant
dim       = 512
pair_dim  = 256      # ← v1 trunk의 token_z와 동일한 bandwidth
depth     = 12       # ← v1 trunk depth와 동일
num_heads = 8
```

- 실제 파라미터 수는 설정에 따라 v1보다 크거나 작을 수 있다. **실행 시점에 측정한 실제 파라미터 수를 기록하고, 그 숫자를 해석 맥락에 포함한다.**
- 동일 시드/데이터/optim으로 Step 5와 같은 짧은 학습 예산으로 돌린다.

해석 규칙:

- **high-bandwidth v2가 v1과 동등** → gap 원인은 pair entry bandwidth. 후속 실험에서는 `pair_dim=256`을 기본으로 간다.
- **high-bandwidth v2도 여전히 뒤쳐짐** → gap 원인은 구조 단순화 (예: trunk 진입 시 outer-product 재초기화 누락, 또는 decoder의 pair-off가 실제로 기여 중이었음). 이 경우엔 v2 구조를 재검토해야 한다 (예: layer 중간에 pair refresh 1회 삽입, 또는 마지막 K layer `pair_dim=0` 도입).
- **high-bandwidth v2가 v1보다 유의하게 나음** → gap 원인은 원본 3-stage 구조의 bottleneck. v2 방향이 더 정당화됨.

이 Step은 선택적이지만, 결과 해석을 흐리게 두고 다음 실험으로 넘어가는 것보다 **한 번 돌려서 원인을 분리하는 비용이 훨씬 싸다**.

### Step 6. 그 다음에 compressed v2 실험

그 후에야 아래를 본다.

- `384 / 128 / 14`
- `448 / 128 / 14`
- `pair_dim` 축소
- `pair_off_last_k`

이 순서여야 실험 해석이 깨끗하다.

---

## 위험 요소와 완화

| 위험 | 설명 | 완화 |
|---|---|---|
| 구조 변경과 용량 변경이 섞임 | 성능 차이 원인 해석이 어려움 | 1차는 capacity-matched v2 사용 |
| 기존 checkpoint 호환성 저하 | v2 state_dict는 v1과 다름 | arch metadata + legacy fallback 유지 |
| inference 복원 실패 | 현재 `args.json` 기대와 실제 저장 경로가 어긋남 | `train.py`에서 metadata를 명시적으로 저장 |
| 테스트 취약성 | 일부 테스트가 stage 구조에 기대고 있음 | invariant test와 arch-specific smoke test 분리 |
| pair 계산 비용 | dense `(B, N, N, pair_dim)`이 메모리를 먹음 | 1차는 `pair_dim=128` 유지, 이후 ablation으로 조정 |

---

## Definition of Done

다음이 만족되면 migration 1차는 완료로 본다.

1. `v1`과 `v2`를 모두 instantiate / load 가능하다.
2. `v2`는 단일 uniform DiT backbone을 사용한다.
3. `forward` 시그니처는 v1과 동일하다.
4. `train.py`가 `args.json`을 자동 저장한다 (v1은 기존 flat schema, v2는 `arch: "v2"` + 중첩 `model_config`). 저장되는 metadata는 **config dataclass의 모든 shape/behavior 필드**를 포함한다 (`delta_e_freq_dim`, `num_elements`, `num_tags` 포함).
5. `inference.py`가 v1/v2 checkpoint를 모두 복원할 수 있다. **Lightning `last.ckpt` 포맷을 직접 로드할 수 있어야 한다** (`state["state_dict"]` + `"model."` prefix stripping). `state["model"]` 구 포맷과 raw state_dict 포맷도 계속 지원한다.
6. **★ 기존에 존재하는 v1 checkpoint + v1 `args.json` (arch 필드 없음) 조합을 코드/파일 수정 없이 로드해서 end-to-end sampling이 통과한다.** 이것이 이번 PR의 **가장 중요한 호환성 계약**이며, 이를 보장하는 regression test 3종(raw / Lightning / 비-default 필드)이 `tests/`에 추가되어 있어야 한다.
7. **★ Resume arch 검사**: 기존 `out_dir`에 v1 run이 있을 때 `--arch v2`로 `train.py`를 실행하면 `args.json`/checkpoint를 덮어쓰기 **전에** fail-fast한다. Regression test 존재.
8. 테스트가 v1/v2 모두 통과한다.
9. capacity-matched v2가 짧은 학습에서 v1 대비 유의하게 나쁘지 않다 (예: 5k step 이내 val loss가 v1 대비 +5% 이내, NaN/Inf 없이 학습 안정).

---

## 후속 실험 순서

v2 baseline이 안정화된 뒤에 아래 순서로 가는 것이 좋다.

1. `384 / 128 / 14` 같은 smaller v2
2. `pair_off_last_k`
3. distance RBF
4. adsorbate-centric cross-attention
5. frozen atom KV-only

즉 **먼저 구조를 단순화하고 baseline을 고정한 뒤, 그 위에 도메인 실험을 얹는 것**이
AdsorbGen 전체 코드베이스에서 가장 좋은 수정 전략이다.
