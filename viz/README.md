# AdsorbGen Replay Viz — 실행 매뉴얼

`replay_viz/ep{N}/`에 캡처된 flow 예측 + UMA relaxation trajectory를 NGL.js 3D
viewer로 보고, 에너지 곡선을 Plotly로 분석하는 로컬 웹 UI.

---

## 사전 확인 (svr8 안에서)

데이터가 캡처돼 있어야 한다. 학습이 진행되면서 replay eval이 끝날 때마다
`runs/<run_name>/replay_viz/ep{N}/`이 생성된다.

```bash
ls /home/minkyu/Cat-bench/runs/full_run_w_replay/replay_viz/ep12/_index.json
```

`_index.json`이 보이면 OK. 안 보이면 아직 replay eval이 한 번도 완료되지
않은 것이다.

각 시스템 폴더 안에는 다음이 있다:
```
sys_XXX/
  x0.pdb           — 초기 placement (NGL 단일 프레임)
  x1_flow.pdb      — flow 모델 예측 (NGL 단일 프레임)
  x1_relaxed.pdb   — UMA 종점 (수렴된 시스템만)
  traj.xyz         — UMA relaxation 전체 trajectory (NGL multi-frame)
  data.npz         — per-step energy + fmax (numpy)
  meta.json        — sid, E_pred, E_gt, fmax_final, status, ...
```

---

## Step 1 (svr8 tmux): backend + frontend 띄우기

`tmux`에 새 창 두 개 만든다 (또는 SSH 두 번 들어가서 따로).

### 창 A — Backend (FastAPI, :8000)

```bash
cd /home/minkyu/Cat-bench
bash viz/run_viz.sh backend
```

정상 출력:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process ...
```

### 창 B — Frontend (Next.js dev, :3000)

```bash
cd /home/minkyu/Cat-bench
bash viz/run_viz.sh frontend
```

정상 출력:
```
▲ Next.js 14.2.21
- Local:        http://localhost:3000
✓ Ready in ~3s
```

> **`bash viz/run_viz.sh install`은 한 번만**: 처음 셋업할 때 `npm install`을
> 실행해 frontend 의존성을 설치한다. 이미 한 번 돌렸으면 다시 할 필요 없음.

### 둘 다 떠있는지 확인

svr8에서:
```bash
ss -tlnp 2>/dev/null | grep -E ":3000|:8000"
```
두 줄 다 LISTEN 떠야 함.

---

## Step 2 (노트북): SSH 포트 포워딩

노트북에서 **새 터미널**을 연다. 평소 svr8에 들어갈 때 쓰는 명령에 `-L` 두
개를 추가한다.

내 SSH config에 `Proxy_svr8` alias가 있는 경우:

```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 Proxy_svr8
```

(SSH config 예시):
```
Host Proxy_svr8
  HostName 59.29.246.29
  User minkyu
  Port 22229
  ProxyCommand ssh ProxyServer -W %h:%p
```

이 명령은 `ProxyCommand`를 자동 적용해서 점프호스트를 통해 svr8에 닿고,
3000/8000 두 포트를 노트북 → svr8로 터널링한다. **이 SSH 세션은 닫지 말고
유지한다.**

---

## Step 3 (노트북): 브라우저 접속

노트북 Chrome (또는 Safari/Firefox) 주소창에:

```
http://localhost:3000
```

`localhost`는 노트북 자기 자신을 가리키지만, Step 2의 SSH 터널 덕분에
3000번 포트로 들어온 트래픽이 svr8의 3000번 포트(=Next.js dev server)로
전달된다.

---

## UI 사용법

화면 구성:

```
┌──────────────────────────────────────────────────────────────┐
│ AdsorbGen Replay Viz                          12 epoch · 32  │
├──────────────────┬───────────────────────────────────────────┤
│ Epoch:    ep12 ▾ │ [메타: sid · E_pred · E_gt · fmax · ...]  │
│                  │ ┌─────────┬─────────┬─────────┐           │
│ Systems (32):    │ │ x_0     │x_1 flow │x_1 relax│           │
│ ▸ sys_003 SUCC   │ │ (3D)    │ (3D)    │ (3D)    │           │
│   sid=1596747    │ └─────────┴─────────┴─────────┘           │
│   64a fmax=0.003 │                                           │
│ ▸ sys_008 ok     │ Relaxation trajectory                     │
│ ▸ sys_014 unconv │ ┌──────────────┬──────────────────┐       │
│ ...              │ │ traj 3D      │ Energy + fmax    │       │
│                  │ │ (scrubber)   │ Plotly           │       │
│                  │ └──────────────┴──────────────────┘       │
└──────────────────┴───────────────────────────────────────────┘
```

- **사이드바**:
  - Epoch 드롭다운 — 여러 epoch이 있으면 선택 가능 (현재는 ep12 하나)
  - 시스템 리스트 — 32개. 상태 배지 색깔:
    - 초록 `SUCCESS`  : E_pred + δ < E_gt (모델이 GT보다 낮은 에너지 찾음)
    - 파랑 `ok`       : 수렴 + anomaly 통과 (단 GT 능가 못함)
    - 노랑 `uma_unconverged` : 500 step 안에 fmax ≤ 0.05 못 도달
    - 빨강 anomaly    : dissociated/desorbed/surface_changed/intercalated/overlap
- **메타 패널** : 선택한 시스템의 sid, ads_id, E_pred, E_gt, improvement,
  fmax, n_steps, status
- **3D 뷰 3개**: x_0 (prior placement), x_1_flow (모델 예측), x_1_relaxed
  (UMA 종점). 마우스 드래그로 회전, 휠로 줌, 우클릭 드래그로 이동.
- **Trajectory + Energy plot**:
  - 좌측: trajectory 3D 뷰 + slider — slider 드래그하면 frame이 바뀜.
    play/pause, skip-to-start/end 버튼.
  - 우측: Energy(좌축, 파랑) + fmax(우축 log scale, 주황) 그래프.
    **그래프를 클릭하면 그 step으로 jump** 한다. 현재 step은 흰 다이아몬드 마커.

### 검증 체크리스트

1. 사이드바에 **32 시스템** 보이는가?
2. `sys_003` 같은 SUCCESS/ok 시스템 클릭 → 3D 뷰 3개 모두 로드되는가?
3. `uma_unconverged` 시스템 클릭 → 우측 패널에 "Not converged" 표시되는가?
4. trajectory 슬라이더 드래그 → 3D 좌표가 부드럽게 바뀌는가?
5. 에너지 곡선이 step에 따라 내려가다 평탄해지는 모양인가?

---

## 종료

- 각 tmux 창에서 `Ctrl+C`
- SSH 터널은 노트북 SSH 세션 종료(`exit` 또는 `Ctrl+D`)

---

## 트러블슈팅

### "VIZ_ROOT not found" / health check 503
backend가 데이터 디렉토리를 찾지 못함. 환경변수로 명시:
```bash
REPLAY_VIZ_ROOT=/home/minkyu/Cat-bench/runs/full_run_w_replay/replay_viz \
  bash viz/run_viz.sh backend
```

### 브라우저에서 계속 로딩만 됨
1. svr8에서 backend/frontend 둘 다 떠있는지: `ss -tlnp | grep -E ":3000|:8000"`
2. 노트북 SSH 터널 살아있는지: 노트북 새 터미널에서
   `curl http://localhost:8000/api/health` → `{"ok":true,...}` 나와야 함
3. 그래도 안 되면 노트북에서 SSH 세션 유지된 상태인지 (창 안 닫혔는지) 확인.

### `ssh: connect to host ... port 22: No route to host`
사설 LAN 주소(`192.168.x.x`)는 같은 네트워크에서만 닿는다. 평소 쓰는 SSH
config alias나 점프호스트 명령으로 가야 한다 (위 Step 2 참조).

### `ssh: connect to host svr8 port 22: Invalid argument`
svr8 안에서 svr8을 ssh 한 케이스. SSH 명령은 **노트북에서** 실행해야 한다.

### NGL 3D 안 뜸
프론트엔드가 `cdn.jsdelivr.net/npm/ngl/dist/ngl.js`에서 NGL을 로드한다.
인터넷 안 되는 환경이면 NGL CDN 접근 가능한지 확인.

### CORS / fetch error
Backend(:8000)와 Frontend(:3000) 둘 다 떠있어야 한다. Frontend
`next.config.mjs`에 `/api/*` → backend로 가는 rewrite가 있어서, 같은
`localhost:3000` 안에서 동작.

### 데이터가 안 바뀜 / 옛 데이터 보임
`replay_viz/ep{N}/`은 **replay 주기마다 전체 교체**된다 (`rotate_viz_dir`
가 이전 epoch 디렉토리 삭제). 다음 replay 완료 후 사이드바 Epoch 드롭다운
새로고침 또는 페이지 새로고침.

---

## 참고

- 데이터 캡처는 `adsorbgen/replay/viz.py`의 `TrajectoryHook`이 담당
  (nvalchemi.dynamics FIRE에 attach). 매 step에서 positions / energy / fmax
  기록.
- Backend 핫 리로드 ON: `viz/backend/main.py` 수정 시 자동 재시작.
- Frontend 핫 리로드 ON (Next.js 기본).
- 데이터 경로 변경:
  ```bash
  REPLAY_VIZ_ROOT=/path/to/other/replay_viz bash viz/run_viz.sh backend
  ```
