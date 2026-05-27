#!/usr/bin/env python
"""Continuous replay daemon (one process per GPU).

What it does, in a loop until a kill signal is detected:
    1. wait for the training run's ``last.ckpt`` to settle and copy it locally
    2. load the model from that snapshot
    3. run one replay eval cycle (flow sample → UMA relax → anomaly/energy
       check) using fairchem's existing ``run_replay_eval`` helper
    4. append successful entries to a sharded append-only ``ReplayStream``
       so the training process can consume them via ``--external-replay-dir``
    5. dump per-cycle metrics + a one-line summary, including NVML GPU util
       statistics measured for the duration of the cycle

Designed to saturate GPU 4-7 (or any subset) by spawning one daemon process
per GPU.  See ``scripts/replay/run_replay_5000x10_8gpu.sh`` for an example
wrapper.

Stopping:
    * Touch ``{stream_dir}/KILL_FLAG`` to make every daemon process exit
      after its current cycle finishes cleanly.
    * Or kill the process group directly (writers flush on signal).
"""
from __future__ import annotations

import argparse
import json
import lmdb
import os
import pickle
import shutil
import signal
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, quantiles
from typing import Optional

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if (_REPO / "adsorbgen").is_dir():
    sys.path.insert(0, str(_REPO))

from adsorbgen.data.dataset import PreprocessedDisplacementDataset  # noqa: E402
from adsorbgen.evaluation.metrics import load_pristine_context  # noqa: E402
from adsorbgen.replay.eval import ReplayEvalConfig, run_replay_eval  # noqa: E402
from adsorbgen.flow import FlowConfig  # noqa: E402
from adsorbgen.models.dit import DiTDenoiserConfig  # noqa: E402
from adsorbgen.models.factory import build_model  # noqa: E402
from adsorbgen.models.dit_v2 import DiTDenoiserV2Config  # noqa: E402
from adsorbgen.replay import ReplayBuffer, ReplayStreamWriter  # noqa: E402


# ---------------------------------------------------------------------------
# NVML util sampler — background thread, low overhead
# ---------------------------------------------------------------------------
class GpuUtilSampler:
    def __init__(self, gpu_index: int = 0, interval_sec: float = 1.0):
        self.gpu_index = int(gpu_index)
        self.interval = float(interval_sec)
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception as exc:  # pragma: no cover - hardware/lib dependent
            print(f"[nvml] disabled: {exc}", flush=True)
            self._pynvml = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                self._samples.append(float(util))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        if self._pynvml is None:
            return
        self._samples.clear()
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self) -> dict:
        if self._th is None:
            return {"available": False, "n_samples": 0}
        self._stop.set()
        self._th.join(timeout=5.0)
        s = list(self._samples)
        if not s:
            return {"available": True, "n_samples": 0}
        s_sorted = sorted(s)
        try:
            qs = quantiles(s_sorted, n=10)
            p10 = qs[0]
            p50 = qs[4]
            p90 = qs[8]
        except Exception:
            p10 = s_sorted[max(0, len(s_sorted) // 10 - 1)]
            p50 = s_sorted[len(s_sorted) // 2]
            p90 = s_sorted[min(len(s_sorted) - 1, (9 * len(s_sorted)) // 10)]
        return {
            "available": True,
            "n_samples": len(s),
            "mean": float(mean(s)),
            "p10": float(p10),
            "p50": float(p50),
            "p90": float(p90),
            "min": float(min(s)),
            "max": float(max(s)),
        }


def _visible_nvml_index(default: int = 0) -> int:
    """Best-effort physical NVML index for a single visible CUDA device."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return int(default)
    first = visible.split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return int(default)


# ---------------------------------------------------------------------------
# Unique-system index across multiple LMDBs
# ---------------------------------------------------------------------------
def _read_raw_sids(lmdb_path: str) -> np.ndarray:
    """Read every entry's sid from an LMDB, indexed by raw entry index."""
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        raw = txn.get(b"length")
        n = int(pickle.loads(raw)) if raw is not None else txn.stat()["entries"]
        sids = np.empty(n, dtype=np.int64)
        for i in range(n):
            sids[i] = int(pickle.loads(txn.get(str(i).encode()))["sid"])
    env.close()
    return sids


def build_unique_system_index(datasets, lmdb_paths, gt_index) -> list:
    """One representative (lmdb_id, dataset_idx) per distinct eligible system_key.

    A 'system' is a gt_index ``system_key``; LMDB rows mapping to the same key
    (or to ineligible / unmapped sids) are collapsed, so replay samples unique
    systems rather than raw data rows.
    """
    uniq: list = []
    seen: set = set()
    for lid, (ds, path) in enumerate(zip(datasets, lmdb_paths)):
        raw_sids = _read_raw_sids(path)
        idx_map = getattr(ds, "_idx_map", None)
        kept = 0
        for di in range(len(ds)):
            raw = int(idx_map[di]) if idx_map is not None else di
            gi = gt_index.get(int(raw_sids[raw]))
            if not (isinstance(gi, dict) and gi.get("eligible")
                    and gi.get("system_key") is not None
                    and (gi.get("E_sys_mean") is not None or gi.get("E_sys_min") is not None)):
                continue
            sk = gi["system_key"]
            if sk in seen:
                continue
            seen.add(sk)
            uniq.append((lid, di))
            kept += 1
        print(f"[daemon]   {Path(path).name}: {len(ds)} rows -> {kept} new unique systems",
              flush=True)
    return uniq


def _aggregate_metrics(ms: list) -> dict:
    """Merge per-LMDB run_replay_eval metrics into one cycle-level dict.

    Strict/relaxed success counts sum across LMDBs because build_unique_system_index
    assigns each unique system_key to exactly one LMDB slice — so per-LMDB
    ``strict_success_systems`` / ``relaxed_success_systems`` sets are disjoint.
    """
    tot_c = sum(m.get("candidates", 0) for m in ms)
    out = {
        "candidates": tot_c,
        "systems_evaluated": sum(m.get("systems_evaluated", 0) for m in ms),
        "n_success": sum(m.get("n_success", 0) for m in ms),
        "n_success_systems": sum(m.get("n_success_systems", 0) for m in ms),
        "n_added_to_buffer": sum(m.get("n_added_to_buffer", 0) for m in ms),
    }
    for k in ("n_relaxed_success_plus_0p1",
              "n_relaxed_success_plus_0p2",
              "n_relaxed_success_plus_0p3",
              "n_relaxed_success_systems_plus_0p1",
              "n_relaxed_success_systems_plus_0p2",
              "n_relaxed_success_systems_plus_0p3"):
        out[k] = sum(m.get(k, 0) for m in ms)
    for rk in ("valid_rate", "dissoc_rate", "desorbed_rate", "surf_changed_rate",
               "intercalated_rate", "overlap_rate", "uma_unconverged_rate"):
        out[rk] = (sum(m.get(rk, 0.0) * m.get("candidates", 0) for m in ms)
                   / max(tot_c, 1))
    return out


# ---------------------------------------------------------------------------
# Ckpt stable copy
# ---------------------------------------------------------------------------
def wait_for_stable_ckpt(
    src: Path,
    dst: Path,
    settle_sec: float = 5.0,
    max_wait_sec: float = 3600.0,
    poll_sec: float = 5.0,
) -> Path:
    """Block until ``src`` exists and its mtime is unchanged for ``settle_sec``,
    then atomic-copy to ``dst``."""
    deadline = time.time() + max_wait_sec
    last_mtime = None
    stable_since: Optional[float] = None
    while time.time() < deadline:
        if not src.exists():
            time.sleep(poll_sec)
            continue
        mt = src.stat().st_mtime
        now = time.time()
        if last_mtime is None or mt != last_mtime:
            last_mtime = mt
            stable_since = now
        elif now - stable_since >= settle_sec:
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            return dst
        time.sleep(poll_sec)
    raise TimeoutError(f"ckpt at {src} never stabilised within {max_wait_sec}s")


# ---------------------------------------------------------------------------
# Model loading (mirrors replay_one_ckpt.py)
# ---------------------------------------------------------------------------
def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    torch.serialization.add_safe_globals(
        [DiTDenoiserConfig, DiTDenoiserV2Config, FlowConfig]
    )
    import sys
    import adsorbgen.models.dit as _dit_mod
    import adsorbgen.models.dit_v2 as _dit_v2_mod
    sys.modules.setdefault("adsorbgen.model", _dit_mod)
    sys.modules.setdefault("adsorbgen.model.dit", _dit_mod)
    sys.modules.setdefault("adsorbgen.model.dit_v2", _dit_v2_mod)
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    model = build_model(hp["model_cfg"])
    sd = ck["state_dict"]
    stripped = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(stripped, strict=False)
    model.adsorbgen_movable_mode = str(hp.get("movable_mode", "surface_ads"))
    model.to(device).eval()
    return model, hp["flow_cfg"]


# ---------------------------------------------------------------------------
# Daemon main loop
# ---------------------------------------------------------------------------
_stop_requested = False


def _scheduled_margin(
    cycle: int,
    initial: float,
    final: Optional[float],
    schedule_cycles: int,
) -> float:
    """Linear interpolation initial→final over ``schedule_cycles`` cycles.
    Off (constant ``initial``) when ``schedule_cycles<=0`` or ``final`` is None."""
    if schedule_cycles <= 0 or final is None:
        return float(initial)
    t = min(max(cycle, 0) / float(schedule_cycles), 1.0)
    return float((1.0 - t) * initial + t * final)


def _on_signal(signum, _frame):  # noqa: D401
    global _stop_requested
    _stop_requested = True
    print(f"[daemon] caught signal {signum}; finishing current cycle then exit",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-run-dir", required=True, type=str,
                   help="training run dir; daemon reads {dir}/last.ckpt")
    p.add_argument("--stream-dir", required=True, type=str,
                   help="root dir for ReplayStream + per-cycle logs (created if missing)")
    p.add_argument("--shard-idx", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=4)

    p.add_argument("--gt-index", required=True, type=str)
    p.add_argument("--train-lmdb", required=True, nargs="+",
                   help="one or more training LMDB paths (full training set is "
                        "5 LMDBs: is2res_train + val_id + 3x val_ood)")
    p.add_argument("--num-systems", type=int, default=200,
                   help="unique systems per shard per cycle "
                        "(total per cycle = num-systems * num-shards)")
    p.add_argument("--num-placements", type=int, default=3)
    p.add_argument("--flow-steps", type=int, default=50)
    p.add_argument("--prior-mode",
                   choices=[
                       "random", "heuristic", "random_heuristic",
                       "harmonic_uniform", "harmonic_centered",
                       "catflow_center_rel",
                   ],
                   default="random_heuristic")
    p.add_argument("--uma-model", default="uma-s-1p1")
    p.add_argument("--uma-fmax", type=float, default=0.05)
    p.add_argument("--uma-max-steps", type=int, default=100)
    p.add_argument("--uma-atom-budget", type=int, default=4000)
    p.add_argument("--flow-batch-size", type=int, default=32)
    p.add_argument("--success-margin", type=float, default=0.05,
                   help="initial success margin (eV); when --success-margin-schedule-cycles>0, "
                        "linearly anneals toward --success-margin-final over that many cycles.")
    p.add_argument("--success-margin-final", type=float, default=None,
                   help="final success margin after annealing. None = no schedule (constant).")
    p.add_argument("--success-margin-schedule-cycles", type=int, default=0,
                   help="number of cycles to linearly anneal success-margin from initial to final. "
                        "0 = no schedule (default).")
    p.add_argument("--overlap-threshold", type=float, default=0.5)
    p.add_argument("--viz-capture-n", type=int, default=0)
    p.add_argument("--e-gt-key", default="",
                   help='override gt_index key for E_gt (e.g. "E_sys_min"); '
                        'empty = default E_sys_mean→E_sys_min fallback')
    p.add_argument("--collect-predictions", action="store_true",
                   help="dump every per-candidate prediction to a pkl per "
                        "cycle (for strict + relaxed reporting); off by default")
    p.add_argument("--use-sde", action="store_true",
                   help="use AtomMOF-style SDE sampling instead of deterministic ODE")
    p.add_argument("--refine-final", action="store_true",
                   help="after sampling, make one extra model call at t=1-eps and use it as x1")

    p.add_argument("--pristine-slabs", type=str, default="")
    p.add_argument("--pristine-index", "--pristine-sid-index",
                   dest="pristine_sid_index", type=str, default="")

    p.add_argument("--chunk-size", type=int, default=64,
                   help="ReplayStreamWriter chunk size (entries per pkl file)")
    p.add_argument("--ckpt-settle-sec", type=float, default=5.0)
    p.add_argument("--ckpt-stale-warn-min", type=float, default=60.0)
    p.add_argument("--ckpt-stale-exit-min", type=float, default=180.0)
    p.add_argument("--max-cycles", type=int, default=0,
                   help="0 = infinite; >0 limits total cycles (testing)")
    p.add_argument("--util-poll-sec", type=float, default=1.0)
    p.add_argument("--system-seed", type=int, default=-1,
                   help="fixed seed for unique-system draw; <0 uses ckpt mtime/cycle seed")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for UMA relax")
    device = torch.device("cuda")

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    stream_root = Path(args.stream_dir)
    stream_root.mkdir(parents=True, exist_ok=True)
    log_root = stream_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    summary_path = stream_root / "summary.jsonl"
    kill_flag = stream_root / "KILL_FLAG"

    last_ckpt_src = Path(args.train_run_dir) / "last.ckpt"
    ckpt_cache = stream_root / f"ckpt_shard{args.shard_idx}.ckpt"

    # gt-index + dataset loaded once and reused across cycles
    with open(args.gt_index, "rb") as f:
        gt_index_by_sid = pickle.load(f)
    print(f"[daemon] gt_index: {len(gt_index_by_sid)} sids", flush=True)

    if args.pristine_slabs:
        prist_pkl = Path(args.pristine_slabs)
        prist_idx = Path(args.pristine_sid_index) if args.pristine_sid_index else None
        load_pristine_context(prist_pkl, prist_idx)
        print(f"[daemon] pristine slab context loaded", flush=True)

    datasets = [
        PreprocessedDisplacementDataset(p, max_samples=None)
        for p in args.train_lmdb
    ]
    print(f"[daemon] loaded {len(datasets)} LMDB(s); building unique-system index...",
          flush=True)
    uniq_index = build_unique_system_index(datasets, args.train_lmdb, gt_index_by_sid)
    n_uniq = len(uniq_index)
    print(f"[daemon] unique systems across all LMDBs: {n_uniq}", flush=True)

    writer = ReplayStreamWriter(stream_root, shard_id=args.shard_idx, chunk_size=args.chunk_size)
    print(f"[daemon] writer ready: {writer.root}", flush=True)

    cycle = 0
    last_seen_ckpt_mtime: Optional[float] = None
    last_ckpt_advance_t = time.time()

    while not _stop_requested:
        if kill_flag.exists():
            print(f"[daemon] KILL_FLAG present; exiting", flush=True)
            break
        if args.max_cycles and cycle >= args.max_cycles:
            print(f"[daemon] reached max_cycles={args.max_cycles}; exiting", flush=True)
            break

        # ---- 1. wait for + copy ckpt
        try:
            wait_for_stable_ckpt(
                last_ckpt_src, ckpt_cache,
                settle_sec=args.ckpt_settle_sec,
                max_wait_sec=args.ckpt_stale_exit_min * 60.0,
            )
        except TimeoutError as exc:
            print(f"[daemon] FATAL: {exc}", flush=True)
            break

        ckpt_mtime = ckpt_cache.stat().st_mtime
        if last_seen_ckpt_mtime is not None and ckpt_mtime == last_seen_ckpt_mtime:
            stale_min = (time.time() - last_ckpt_advance_t) / 60.0
            if stale_min >= args.ckpt_stale_exit_min:
                print(f"[daemon] ckpt stale for {stale_min:.1f}min ≥ exit; quitting",
                      flush=True)
                break
            if stale_min >= args.ckpt_stale_warn_min:
                print(f"[daemon] WARN: ckpt stale for {stale_min:.1f}min (no training progress?)",
                      flush=True)
        else:
            last_seen_ckpt_mtime = ckpt_mtime
            last_ckpt_advance_t = time.time()

        # ---- 2. load model
        t_load = time.time()
        model, flow_cfg = load_model_from_ckpt(ckpt_cache, device)
        print(f"[daemon] cycle {cycle}: ckpt loaded in {time.time()-t_load:.1f}s", flush=True)

        # ---- 3. sample unique systems for this cycle (shared seed across shards
        #         => the 8 shards partition the same draw with no overlap/gap)
        if args.system_seed >= 0:
            pick_seed = (int(args.system_seed) + int(cycle)) & 0xFFFF_FFFF
        else:
            pick_seed = hash((cycle, int(ckpt_mtime))) & 0xFFFF_FFFF
        rng = np.random.default_rng(seed=pick_seed)
        n_pick = min(args.num_systems * args.num_shards, n_uniq)
        picked = rng.choice(n_uniq, size=n_pick, replace=False)
        my_picked = picked[args.shard_idx::args.num_shards]
        # group this shard's unique systems by source LMDB
        by_lmdb: dict = defaultdict(list)
        for u in my_picked.tolist():
            lid, di = uniq_index[u]
            by_lmdb[lid].append(di)

        cfg = ReplayEvalConfig(
            prior_mode=args.prior_mode,
            num_systems=int(len(my_picked)),
            num_placements=args.num_placements,
            flow_steps=args.flow_steps,
            uma_model=args.uma_model,
            uma_fmax=args.uma_fmax,
            uma_max_steps=args.uma_max_steps,
            overlap_threshold=args.overlap_threshold,
            success_margin=_scheduled_margin(
                cycle, args.success_margin,
                args.success_margin_final,
                args.success_margin_schedule_cycles,
            ),
            device="cuda",
            flow_batch_size=args.flow_batch_size,
            uma_atom_budget=args.uma_atom_budget,
            viz_capture_n=args.viz_capture_n,
            viz_root="",
            e_gt_key=args.e_gt_key,
            collect_predictions=args.collect_predictions,
            use_sde=args.use_sde,
            refine_final=args.refine_final,
        )

        cycle_buffer = ReplayBuffer(mode="append", per_system_cap=1_000_000,
                                    global_cap=1_000_000)

        # ---- 4. sample utilisation while running the cycle
        sampler = GpuUtilSampler(
            gpu_index=_visible_nvml_index(0),
            interval_sec=args.util_poll_sec,
        )
        sampler.start()
        t_cycle = time.time()
        try:
            per_lmdb = []
            for lid, di_list in by_lmdb.items():
                per_lmdb.append(run_replay_eval(
                    model=model,
                    dataset=datasets[lid],
                    gt_index_by_sid=gt_index_by_sid,
                    buffer=cycle_buffer,
                    cfg=cfg,
                    flow_cfg=flow_cfg,
                    epoch=cycle,
                    sys_indices_override=np.asarray(di_list, dtype=np.int64),
                ))
            metrics = _aggregate_metrics(per_lmdb)
        finally:
            util_stats = sampler.stop()
        elapsed = time.time() - t_cycle

        # ---- 5. dump successful entries to the shared stream
        n_streamed = 0
        success_entries = []
        for entry in cycle_buffer.iter_entries():
            writer.append(entry)
            n_streamed += 1
            success_entries.append({
                "sid": int(entry.sid),
                "ads_id": int(entry.ads_id),
                "system_key": list(entry.system_key),
                "E_sys_pred": float(entry.E_sys_pred),
                "E_sys_gt": float(entry.E_sys_gt),
                "improvement": float(entry.improvement),
                "epoch_added": int(entry.epoch_added),
                "source_placement": str(entry.source_placement),
            })
        writer.flush()

        # ---- 6. logging
        cycle_metrics = {
            "cycle": cycle,
            "shard_idx": args.shard_idx,
            "num_shards": args.num_shards,
            "ckpt_mtime": ckpt_mtime,
            "ckpt_path": str(last_ckpt_src),
            "elapsed_sec": elapsed,
            "n_systems": int(metrics.get("systems_evaluated", 0)),
            "candidates": int(metrics.get("candidates", 0)),
            "n_success": int(metrics.get("n_success", 0)),
            "n_added": int(metrics.get("n_added_to_buffer", 0)),
            "n_streamed": n_streamed,
            "success_entries": success_entries,
            "gpu_util": util_stats,
        }
        # also forward any extra keys returned by run_replay_eval
        for k, v in metrics.items():
            if k in cycle_metrics:
                continue
            if isinstance(v, (int, float)):
                cycle_metrics[k] = v

        per_cycle = log_root / f"cycle_{cycle:06d}_shard{args.shard_idx}.json"
        per_cycle.write_text(json.dumps(cycle_metrics, indent=2))

        if args.collect_predictions:
            all_preds = []
            for m in per_lmdb:
                all_preds.extend(m.get("predictions", []))
            pred_path = log_root / f"cycle_{cycle:06d}_shard{args.shard_idx}_predictions.pkl"
            with open(pred_path, "wb") as f:
                pickle.dump(all_preds, f)
            print(f"[daemon] cycle {cycle}: wrote {len(all_preds)} predictions "
                  f"to {pred_path.name}", flush=True)

        # one-line summary
        with open(summary_path, "a") as f:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps({
                    "cycle": cycle,
                    "shard": args.shard_idx,
                    "elapsed_sec": elapsed,
                    "n_success": cycle_metrics["n_success"],
                    "n_streamed": n_streamed,
                    "gpu_util_mean": util_stats.get("mean"),
                    "gpu_util_p10": util_stats.get("p10"),
                    "ckpt_mtime": ckpt_mtime,
                }) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        util_summary = (
            f"util={util_stats.get('mean',float('nan')):.0f}% (p10={util_stats.get('p10',float('nan')):.0f}%)"
            if util_stats.get("available") and util_stats.get("n_samples") else "util=NA"
        )
        print(
            f"[daemon] cycle {cycle} done: "
            f"systems={cycle_metrics['n_systems']} cand={cycle_metrics['candidates']} "
            f"succ={cycle_metrics['n_success']} streamed={n_streamed} "
            f"elapsed={elapsed:.0f}s {util_summary}",
            flush=True,
        )

        cycle += 1
        # free GPU memory between cycles so the next model load is clean
        del model
        torch.cuda.empty_cache()

    writer.flush()
    print(f"[daemon] shutdown after {cycle} cycle(s)", flush=True)


if __name__ == "__main__":
    main()
