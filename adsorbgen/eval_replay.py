"""Replay evaluation: inference → UMA relax → anomaly filter → energy check.

Entry point is ``run_replay_eval`` which populates a ReplayBuffer with
successful entries and logs per-eval statistics.

Conceptual flow per eligible system:
  for k in K placements:
    x0 = fairchem placement(slab, ads, mode=prior_mode)
    x1_pred = flow.euler_sample(model, x0, ...)
    atoms_relaxed = UMA_relax(atoms_from(x1_pred))
    if passes anomaly AND E_sys_pred + δ < E_sys_gt:
        add_to_buffer(entry)

UMA relaxation is parallelised via ``fast_dynamics`` + ``nvalchemi`` batched
FIRE on a single GPU. Flow sampling is batched through the trained model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from ase import Atoms
from tqdm.auto import tqdm

from adsorbgen.dataset import (
    PlacementPriorDataset, PreprocessedDisplacementDataset, collate_displacement,
)
from adsorbgen.flow import FlowConfig, euler_sample
from adsorbgen.replay import ReplayBuffer, ReplayEntry
from adsorbgen.replay_viz import (
    FixedAtomsHook, TrajectoryHook, reference_center_translation,
    rotate_viz_dir,
    save_structure_pdb, save_trajectory_xyz, save_trajectory_pdb,
    save_traj_npz, save_meta_json, write_index,
)


def _atoms_from_batch(batch: Dict[str, torch.Tensor], i: int, pos: torch.Tensor) -> Atoms:
    n = int(batch["pad_mask"][i].sum().item())
    tags_i = batch["tags"][i, :n].cpu().numpy().astype(int)
    cell = batch["cell"][i].cpu().numpy()
    if cell.ndim == 3:
        cell = cell[0]
    return Atoms(
        numbers=batch["atomic_numbers"][i, :n].cpu().numpy().astype(int),
        positions=pos[i, :n].detach().cpu().numpy().astype(np.float64),
        cell=cell, pbc=True, tags=tags_i.tolist(),
    )


def _passes_anomaly(
    pos_ref: np.ndarray,
    pos_pred: np.ndarray,
    pos_gt: np.ndarray,
    atomic_numbers: np.ndarray,
    tags: np.ndarray,
    cell: np.ndarray,
    sid: int,
    ads_id: int = -1,
) -> tuple:
    """Validation-equivalent 5-axis anomaly check.

    Delegates to ``adsorbgen.eval._score_record_anomaly`` so replay shares the
    *same* anomaly definition as the sample_eval/validation pipeline:
      * overlap:  MIC min-pair distance < 0.5 Å (``OVERLAP_MIN_DIST_A``)
      * dissoc / desorbed / intercalated / surf_changed:
            fairchem ``DetectTrajAnomaly`` (not phase3's reimplementation).
            ``ads_id`` is forwarded so the dissoc bond-graph check uses the
            canonical gas-phase adsorbate geometry as init reference, instead
            of the (bond-graph-invalid) prior x_0 — matching validation.
      * surf_changed reference: pristine relaxed slab (sid lookup) when
            ``load_pristine_context`` has been called by the caller; otherwise
            falls back to ``pos_gt[tags != 2]``.

    Args correspond to validation's ``record`` dict:
        pos_ref:  (N, 3) placement prior x_0 (= init_atoms positions)
        pos_pred: (N, 3) the geometry to evaluate (= UMA-relaxed positions)
        pos_gt:   (N, 3) ground-truth relaxed positions from the LMDB sample
                  (used as final_slab fallback when pristine isn't loaded)
        cell:     (3, 3)

    Returns (passes: bool, reason: str). Reason ∈ {"ok", "dissociated",
    "desorbed", "surface_changed", "intercalated", "overlap"}.
    """
    from adsorbgen.eval import _score_record_anomaly
    record = {
        "sid": int(sid),
        "ads_id": int(ads_id),
        "pos_ref": torch.as_tensor(pos_ref, dtype=torch.float32),
        "pos_pred": torch.as_tensor(pos_pred, dtype=torch.float32),
        "pos_gt": torch.as_tensor(pos_gt, dtype=torch.float32),
        "atomic_numbers": torch.as_tensor(atomic_numbers, dtype=torch.long),
        "tags": torch.as_tensor(tags, dtype=torch.long),
        "cell": torch.as_tensor(cell, dtype=torch.float32),
    }
    res = _score_record_anomaly(record)
    if res.get("error") == "nonfinite_pred":
        return False, "overlap"
    if res["has_dissoc"]:        return False, "dissociated"
    if res["has_desorbed"]:      return False, "desorbed"
    if res["has_surf_changed"]:  return False, "surface_changed"
    if res["has_intercalated"]:  return False, "intercalated"
    if res["has_overlap"]:       return False, "overlap"
    return True, "ok"


@dataclass
class ReplayEvalConfig:
    prior_mode: str = "random_heuristic"
    num_systems: int = 500
    num_placements: int = 3
    flow_steps: int = 50
    uma_model: str = "uma-s-1p1"
    uma_task: str = "oc20"
    uma_fmax: float = 0.05
    uma_max_steps: int = 100
    overlap_threshold: float = 0.5
    success_margin: float = 0.05   # eV
    device: str = "cuda"
    # --- batching knobs for fast_dynamics path ---
    flow_batch_size: int = 32          # flow euler_sample batch (samples / fwd)
    uma_atom_budget: int = 4000        # target atoms per UMA FIRE batch
    uma_fire_dt: float = 0.02          # FIRE timestep
    # --- visualization capture ---
    viz_capture_n: int = 32            # systems to save viz artifacts for each cycle
    viz_root: str = ""                 # dir to store replay_viz/ep{N}/ (empty → disabled)


def _chunk_by_atom_budget(items: List[dict], budget: int) -> List[List[dict]]:
    """Pack items greedily into chunks s.t. sum(item['n_atoms']) <= budget."""
    chunks: List[List[dict]] = []
    cur: List[dict] = []
    cur_n = 0
    for it in items:
        n = int(it["n_atoms"])
        if cur and cur_n + n > budget:
            chunks.append(cur)
            cur, cur_n = [], 0
        cur.append(it)
        cur_n += n
    if cur:
        chunks.append(cur)
    return chunks


def _model_cfg(model):
    m = model
    while hasattr(m, "module"):
        m = m.module
    return getattr(m, "cfg", None)


@torch.no_grad()
def run_replay_eval(
    model,
    dataset: PreprocessedDisplacementDataset,  # non-placement base (for sid→sample)
    gt_index_by_sid: dict,                     # from phase 0c output
    buffer: ReplayBuffer,
    cfg: ReplayEvalConfig,
    flow_cfg: FlowConfig,
    epoch: int,
    logger=None,
    sys_indices_override: Optional["np.ndarray"] = None,
) -> Dict:
    """Run one eval pass; mutate buffer in place; return metrics dict.

    Pipeline:
      1. Gather eligible (sys_idx, k) pairs with fresh placement samples.
      2. Batched flow euler_sample → predicted x_1 positions.
      3. Batched UMA FIRE relaxation (nvalchemi) with fmax convergence.
      4. Per-system anomaly filter + energy check → ReplayBuffer.
    """
    from fast_dynamics import UMAWrapper, prepare_batch_for_dynamics
    from nvalchemi.data import AtomicData as NVAtomicData
    from nvalchemi.data import Batch as NVBatch
    from nvalchemi.dynamics import FIRE, ConvergenceHook

    model.eval()
    device = torch.device(cfg.device)
    use_ads_ref = bool(getattr(_model_cfg(model), "use_ads_ref_pos", False))

    # --- One UMA wrapper shared across all batches ---
    uma = UMAWrapper.from_checkpoint(
        cfg.uma_model, task_name=cfg.uma_task, device=device,
    )

    # --- Draw system indices ---
    n_total = len(dataset)
    if sys_indices_override is not None:
        sys_indices = np.asarray(sys_indices_override, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed=epoch)
        sys_indices = rng.choice(n_total, size=min(cfg.num_systems, n_total), replace=False)

    placement_ds = PlacementPriorDataset(
        dataset.lmdb_path,
        prior_mode=cfg.prior_mode,
        max_samples=n_total,
        provide_ads_ref_pos=use_ads_ref,
    )

    # --- Build work list: one entry per (eligible sys_idx, k placement) ---
    work: List[dict] = []
    for idx in sys_indices:
        base_sample = dataset[int(idx)]
        sid = int(base_sample["sid"].item())
        gt_info = gt_index_by_sid.get(sid)
        if gt_info is None or not gt_info.get("eligible"):
            continue
        # Prefer the oc20 rebuilt mean reference when present. For a given
        # unique system, E_slab and E_gas are constants, so comparing E_sys to
        # the group's mean E_sys is equivalent to comparing E_ads to mean E_ads.
        # Fall back to older min-based indexes for backward compatibility.
        E_gt = gt_info.get("E_sys_mean", gt_info.get("E_sys_min"))
        if E_gt is None:
            continue
        system_key = gt_info["system_key"]
        for k in range(cfg.num_placements):
            s = placement_ds[int(idx)]
            work.append({
                "sample": s,
                "sid": sid,
                "ads_id": int(s["ads_id"].item()) if "ads_id" in s else 0,
                "system_key": system_key,
                "E_gt": float(E_gt),
            })

    t0 = time.time()
    total_candidates = len(work)
    if total_candidates == 0:
        return {
            "epoch": epoch, "systems_evaluated": 0, "candidates": 0,
            "n_success": 0, "n_added_to_buffer": 0,
            "buffer_size": len(buffer), "buffer_n_systems": buffer.n_systems(),
            "elapsed_sec": 0.0,
        }

    # --- PHASE 1: batched flow euler_sample → predicted positions ---
    predictions: List[dict] = []  # per-sample dicts with atoms_init + pred_pos + meta
    FB = max(1, int(cfg.flow_batch_size))
    flow_pbar = tqdm(
        range(0, len(work), FB),
        desc=f"[replay ep{epoch+1}] flow sample",
        unit="batch", dynamic_ncols=True, leave=True,
    )
    for start in flow_pbar:
        chunk = work[start:start + FB]
        samples = [w["sample"] for w in chunk]
        batch = collate_displacement(samples)
        batch = {k_: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k_, v in batch.items()}

        def fwd(x_t, t, _b=batch):
            extra = {}
            if use_ads_ref:
                extra["ads_ref_pos"] = _b["ads_ref_pos"]
            return model(
                pos=_b["pos"], x_t=x_t, t=t,
                atomic_numbers=_b["atomic_numbers"], tags=_b["tags"],
                movable_mask=_b["movable_mask"],
                pad_mask=_b["pad_mask"], cell=_b["cell"],
                **extra,
            )

        # Capture the flow Euler trajectory when viz is enabled (deterministic
        # ODE — exactly the inference path, just also kept for rendering).
        _capture_flow = bool(cfg.viz_root)
        _es = euler_sample(
            fwd, batch["pos"],
            batch["movable_mask"], batch["pad_mask"], flow_cfg,
            num_steps=cfg.flow_steps,
            return_trajectory=_capture_flow,
        )
        if _capture_flow:
            x_out, flow_traj_b = _es["x_out"], _es["x_trajectory"]
        else:
            x_out, flow_traj_b = _es, None

        # Extract each sample's predictions (trim padding)
        for i, w in enumerate(chunk):
            n = int(batch["pad_mask"][i].sum().item())
            tags_i = batch["tags"][i, :n].detach().cpu().numpy().astype(np.int64)
            numbers_i = batch["atomic_numbers"][i, :n].detach().cpu().numpy().astype(np.int64)
            cell_i = batch["cell"][i].detach().cpu().numpy()
            if cell_i.ndim == 3:
                cell_i = cell_i[0]
            cell_i = cell_i.astype(np.float32)
            pos_init_i = batch["pos"][i, :n].detach().cpu().numpy().astype(np.float64)
            pos_pred_i = x_out[i, :n].detach().cpu().numpy().astype(np.float64)
            pos_relaxed_i = batch["pos_relaxed"][i, :n].detach().cpu().numpy().astype(np.float64)
            fixed_i = batch["fixed"][i, :n].detach().cpu().numpy().astype(np.int64)

            atoms_init = Atoms(
                numbers=numbers_i, positions=pos_init_i,
                cell=cell_i, pbc=True, tags=tags_i.tolist(),
            )
            predictions.append({
                **w,
                "_global_idx": len(predictions),   # track position across chunking
                "n_atoms": int(n),
                "numbers": numbers_i,
                "tags": tags_i,
                "fixed": fixed_i,
                "cell": cell_i,
                "atoms_init": atoms_init,
                "pos_pred": pos_pred_i,
                "flow_traj": (
                    flow_traj_b[:, i, :n, :].detach().cpu().numpy().astype(np.float32)
                    if flow_traj_b is not None else None
                ),
                "pos_relaxed": pos_relaxed_i,
            })

    # --- Viz capture setup (before chunking so indices are stable) ---
    # Strategy: pick ``viz_capture_n`` UNIQUE sids; capture trajectory for ALL
    # K placements of each chosen sid; after Phase 2 select the placement with
    # the lowest E_pred per sid and write only that "winner" to disk.
    viz_set: set = set()                          # global_idx values to hook-capture
    viz_ep_dir: Optional[Path] = None
    viz_entries: List[dict] = []                  # for _index.json
    viz_offsets: Dict[int, np.ndarray] = {}       # global_idx → centering translation
    viz_traj_data: Dict[int, dict] = {}           # global_idx → traj_hook.trajectories[i]
    viz_results: Dict[int, dict] = {}             # global_idx → per-system result dict
    if cfg.viz_root and cfg.viz_capture_n > 0 and predictions:
        viz_root_p = Path(cfg.viz_root)
        viz_root_p.mkdir(parents=True, exist_ok=True)
        rotate_viz_dir(viz_root_p, epoch + 1)
        viz_ep_dir = viz_root_p / f"ep{epoch + 1}"
        viz_ep_dir.mkdir(parents=True, exist_ok=True)

        # Group predictions by sid → all K placements of one system are siblings.
        sid_to_globals: Dict[int, List[int]] = {}
        for g, pp in enumerate(predictions):
            sid_to_globals.setdefault(int(pp["sid"]), []).append(g)
        eligible_sids = sorted(sid_to_globals.keys())
        # Pick up to ``viz_capture_n`` unique sids deterministically.
        n_viz = min(cfg.viz_capture_n, len(eligible_sids))
        rng_v = np.random.default_rng(seed=epoch)
        chosen_idx = rng_v.choice(len(eligible_sids), size=n_viz, replace=False)
        chosen_sids = {eligible_sids[i] for i in chosen_idx}
        for sid in chosen_sids:
            for g in sid_to_globals[sid]:
                viz_set.add(g)
                p = predictions[g]
                viz_offsets[g] = reference_center_translation(
                    p["atoms_init"].get_positions(), p["cell"],
                )
        # Files are NOT saved yet — winners are picked after Phase 2.

    # --- PHASE 2: batched UMA FIRE relaxation ---
    n_success = n_added = n_candidates_valid = 0
    strict_success_systems: set = set()
    relaxed_success_candidates = {0.1: 0, 0.2: 0, 0.3: 0}
    relaxed_success_systems = {0.1: set(), 0.2: set(), 0.3: set()}
    n_dissoc = n_desorbed = n_surf_changed = n_intercalated = n_overlap = 0
    n_uma_unconverged = 0

    uma_chunks = _chunk_by_atom_budget(predictions, cfg.uma_atom_budget)
    uma_pbar = tqdm(
        uma_chunks,
        desc=f"[replay ep{epoch+1}] UMA relax",
        unit="chunk", dynamic_ncols=True, leave=True,
    )
    for chunk in uma_pbar:
        uma_pbar.set_postfix(
            sys=len(chunk),
            atoms=sum(int(p["n_atoms"]) for p in chunk),
            added=n_added,
            succ=n_success,
            unconv=n_uma_unconverged,
        )
        # Build nvalchemi AtomicData per system (positions = flow-predicted x_1)
        data_list = []
        for p in chunk:
            ad = NVAtomicData(
                positions=torch.as_tensor(p["pos_pred"], dtype=torch.float32, device=device),
                atomic_numbers=torch.as_tensor(p["numbers"], dtype=torch.long, device=device),
                cell=torch.as_tensor(p["cell"], dtype=torch.float32, device=device).reshape(1, 3, 3),
                pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
            )
            data_list.append(ad)
        nvbatch = NVBatch.from_data_list(data_list)
        prepare_batch_for_dynamics(nvbatch)

        # Identify viz targets that live in this chunk (by _global_idx)
        local_viz_indices: List[int] = []
        global_for_local: Dict[int, int] = {}   # local idx → global idx in predictions
        if viz_set:
            for local_i, p in enumerate(chunk):
                g = p["_global_idx"]
                if g in viz_set:
                    local_viz_indices.append(local_i)
                    global_for_local[local_i] = g

        # Build flat fixed-atom mask over the whole chunk (nvalchemi node order
        # is sys0_atoms ++ sys1_atoms ++ ...). LMDB ``fixed`` is the OC20
        # canonical mask; fall back to ``tags == 0`` (subsurface bulk) if
        # ``fixed`` happens to be all-zero for a system.
        fixed_parts = []
        for p in chunk:
            fm = np.asarray(p["fixed"], dtype=bool)
            if not fm.any():
                fm = np.asarray(p["tags"]) == 0
            fixed_parts.append(fm)
        fixed_mask_np = np.concatenate(fixed_parts) if fixed_parts else np.zeros(0, dtype=bool)
        fixed_mask = torch.as_tensor(fixed_mask_np, dtype=torch.bool, device=device)

        fire = FIRE(
            uma,
            dt=cfg.uma_fire_dt,
            n_steps=cfg.uma_max_steps,
            convergence_hook=ConvergenceHook.from_fmax(cfg.uma_fmax),
        )
        # Always register FixedAtomsHook so bulk doesn't drift during relax.
        if fixed_mask.any():
            fire.register_hook(FixedAtomsHook(fixed_mask))

        traj_hook = None
        if local_viz_indices:
            traj_hook = TrajectoryHook(local_viz_indices)
            fire.register_hook(traj_hook)

        fire.run(nvbatch)

        # Extract per-system relaxed state
        ptr = nvbatch.batch_ptr.long().tolist()
        energies = nvbatch.energy.detach().squeeze(-1).float().cpu().tolist()
        forces = nvbatch.forces.detach()
        positions_out = nvbatch.positions.detach()

        chunk_results: Dict[int, dict] = {}   # local_i -> per-system summary for viz

        for i, p in enumerate(chunk):
            start, end = ptr[i], ptr[i + 1]
            fmax_i = float(forces[start:end].norm(dim=-1).max().item())
            converged = bool(fmax_i <= cfg.uma_fmax)
            relaxed_pos = positions_out[start:end].cpu().numpy().astype(np.float32)
            E_pred = float(energies[i])

            status = "ok"
            success = False

            if not converged:
                n_uma_unconverged += 1
                status = "uma_unconverged"
            else:
                passed, reason = _passes_anomaly(
                    pos_ref=p["atoms_init"].get_positions(),
                    pos_pred=relaxed_pos.astype(np.float64),
                    pos_gt=p["pos_relaxed"],
                    atomic_numbers=p["numbers"],
                    tags=p["tags"],
                    cell=p["cell"],
                    sid=p["sid"],
                    ads_id=int(p["ads_id"]),
                )
                if not passed:
                    if reason == "dissociated":       n_dissoc += 1
                    elif reason == "desorbed":         n_desorbed += 1
                    elif reason == "surface_changed":  n_surf_changed += 1
                    elif reason == "intercalated":     n_intercalated += 1
                    elif reason == "overlap":          n_overlap += 1
                    status = reason
                else:
                    n_candidates_valid += 1
                    sys_key = tuple(p["system_key"])
                    for tol in relaxed_success_candidates:
                        if E_pred < p["E_gt"] + tol:
                            relaxed_success_candidates[tol] += 1
                            relaxed_success_systems[tol].add(sys_key)
                    if E_pred + cfg.success_margin < p["E_gt"]:
                        n_success += 1
                        strict_success_systems.add(sys_key)
                        success = True
                        improvement = p["E_gt"] - E_pred
                        entry = ReplayEntry(
                            system_key=tuple(p["system_key"]),
                            sid=p["sid"],
                            ads_id=p["ads_id"],
                            pos_relaxed=relaxed_pos,
                            tags=p["tags"],
                            atomic_numbers=p["numbers"],
                            fixed=p["fixed"],
                            cell=p["cell"],
                            E_sys_pred=E_pred,
                            E_sys_gt=float(p["E_gt"]),
                            improvement=float(improvement),
                            epoch_added=epoch,
                            source_placement=cfg.prior_mode,
                        )
                        if buffer.add(entry):
                            n_added += 1

            chunk_results[i] = {
                "fmax": fmax_i,
                "converged": converged,
                "relaxed_pos": relaxed_pos,
                "E_pred": E_pred,
                "status": status,
                "success": success,
            }

        # --- viz: stash trajectory + per-system result for winner-selection ---
        if traj_hook is not None:
            for local_i, global_i in global_for_local.items():
                viz_traj_data[global_i] = traj_hook.trajectories[local_i]
        # Also stash per-system results for ALL viz targets in this chunk
        # (whether their traj was captured or not — failed FIRE skips traj_hook).
        for local_i, p in enumerate(chunk):
            g = p["_global_idx"]
            if g in viz_set and local_i in chunk_results:
                viz_results[g] = chunk_results[local_i]

    # --- Finalize viz: save winner-per-sid, AND every success placement ---
    # Per-sid winner shows the best representative of each captured sid (even
    # if no success). Every success placement is saved unconditionally so the
    # web UI never hides a hit behind another hit on the same system.
    if viz_ep_dir is not None and viz_set:
        winners_by_sid: Dict[int, int] = {}
        for g in viz_set:
            r = viz_results.get(g)
            if r is None:
                continue
            sid = int(predictions[g]["sid"])
            cur = winners_by_sid.get(sid)
            if cur is None or r["E_pred"] < viz_results[cur]["E_pred"]:
                winners_by_sid[sid] = g

        # Union: winners (one per sid) + every success placement in viz_set.
        to_save: set = set(winners_by_sid.values())
        for g in viz_set:
            r = viz_results.get(g)
            if r is not None and r.get("success"):
                to_save.add(g)

        for g in sorted(to_save):
            sid = int(predictions[g]["sid"])
            p = predictions[g]
            r = viz_results[g]
            offset = viz_offsets.get(g)
            sys_dir = viz_ep_dir / f"sys_{g:03d}"
            sys_dir.mkdir(exist_ok=True)
            save_structure_pdb(
                p["numbers"], p["atoms_init"].get_positions(),
                p["cell"], p["tags"], sys_dir / "x0.pdb", offset=offset,
            )
            save_structure_pdb(
                p["numbers"], p["pos_pred"],
                p["cell"], p["tags"], sys_dir / "x1_flow.pdb", offset=offset,
            )
            ft = p.get("flow_traj")
            if ft is not None:
                save_trajectory_xyz(
                    p["numbers"], ft, p["cell"], p["tags"],
                    sys_dir / "flow_traj.xyz", offset=offset,
                )
                save_trajectory_pdb(
                    p["numbers"], ft, p["cell"], p["tags"],
                    sys_dir / "flow_traj.pdb", offset=offset,
                )
            if r["converged"]:
                save_structure_pdb(
                    p["numbers"], r["relaxed_pos"],
                    p["cell"], p["tags"], sys_dir / "x1_relaxed.pdb",
                    offset=offset,
                )
            td = viz_traj_data.get(g)
            n_steps = 0
            if td is not None:
                save_trajectory_xyz(
                    p["numbers"], td["positions"],
                    p["cell"], p["tags"], sys_dir / "traj.xyz", offset=offset,
                )
                save_trajectory_pdb(
                    p["numbers"], td["positions"],
                    p["cell"], p["tags"], sys_dir / "traj.pdb", offset=offset,
                )
                save_traj_npz(td, sys_dir / "data.npz")
                n_steps = int(len(td["positions"]))
            meta = {
                "global_idx": int(g),
                "sid": int(p["sid"]),
                "ads_id": int(p["ads_id"]),
                "n_atoms": int(p["n_atoms"]),
                "n_steps": n_steps,
                "E_pred": float(r["E_pred"]),
                "E_gt": float(p["E_gt"]),
                "improvement": float(p["E_gt"] - r["E_pred"]),
                "fmax_final": float(r["fmax"]),
                "converged": bool(r["converged"]),
                "status": str(r["status"]),
                "success": bool(r["success"]),
                "n_placements_tried": int(len([gg for gg in viz_set if predictions[gg]["sid"] == sid])),
            }
            save_meta_json(meta, sys_dir / "meta.json")
            viz_entries.append(meta)

        write_index(viz_ep_dir, viz_entries)

    # --- Free GPU memory: drop UMA + chunk tensors before returning. ---
    # ``fire`` holds a reference to ``uma``, so deleting ``uma`` alone is a
    # no-op. Drop FIRE and the last chunk's NVBatch tensors first. After the
    # `for chunk in uma_pbar:` loop runs at least once, all of these names are
    # bound — assert so the cleanup contract is explicit.
    assert "fire" in locals() and "nvbatch" in locals(), (
        "expected at least one chunk to have been processed before cleanup"
    )
    del fire
    del nvbatch
    del forces
    del positions_out
    del traj_hook
    del fixed_mask
    del uma
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    denom = max(total_candidates, 1)
    metrics = {
        "epoch": epoch,
        "systems_evaluated": len(sys_indices),
        "candidates": total_candidates,
        "n_success": n_success,
        "n_success_systems": len(strict_success_systems),
        "n_relaxed_success_plus_0p1": relaxed_success_candidates[0.1],
        "n_relaxed_success_plus_0p2": relaxed_success_candidates[0.2],
        "n_relaxed_success_plus_0p3": relaxed_success_candidates[0.3],
        "n_relaxed_success_systems_plus_0p1": len(relaxed_success_systems[0.1]),
        "n_relaxed_success_systems_plus_0p2": len(relaxed_success_systems[0.2]),
        "n_relaxed_success_systems_plus_0p3": len(relaxed_success_systems[0.3]),
        "n_added_to_buffer": n_added,
        "buffer_size": len(buffer),
        "buffer_n_systems": buffer.n_systems(),
        "elapsed_sec": elapsed,
        "valid_rate": n_candidates_valid / denom,
        "dissoc_rate": n_dissoc / denom,
        "desorbed_rate": n_desorbed / denom,
        "surf_changed_rate": n_surf_changed / denom,
        "intercalated_rate": n_intercalated / denom,
        "overlap_rate": n_overlap / denom,
        "uma_unconverged_rate": n_uma_unconverged / denom,
    }
    if logger is not None:
        logger(metrics)
    return metrics


class ReplayScheduler:
    """Monitors per-eval new_samples history and scales eval budget up.

    Trigger (OR): 5 evals done, OR 3 consecutive evals within ±30% plateau.
    """

    def __init__(self, initial: ReplayEvalConfig, scaled: ReplayEvalConfig):
        self.initial = initial
        self.scaled = scaled
        self.history: List[int] = []
        self.scaled_up = False

    def should_scale(self) -> bool:
        if self.scaled_up:
            return False
        if len(self.history) >= 5:
            return True
        if len(self.history) >= 3:
            last3 = self.history[-3:]
            mean = sum(last3) / 3
            if max(abs(x - mean) for x in last3) / max(mean, 1) <= 0.3:
                return True
        return False

    def current_cfg(self) -> ReplayEvalConfig:
        return self.scaled if self.scaled_up else self.initial

    def record(self, n_added: int):
        self.history.append(int(n_added))
        if self.should_scale():
            self.scaled_up = True
