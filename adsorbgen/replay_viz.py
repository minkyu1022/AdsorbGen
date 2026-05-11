"""Visualization artifact capture for replay eval.

Captures per-step UMA relaxation trajectories for a subset of systems in
each replay eval cycle, along with structural snapshots (x_0 prior, x_1_flow
prediction, x_1_relaxed final) and metadata. Intended to be served by a
separate web UI (AMD-style, FastAPI + Next.js/NGL.js).

Output layout (one directory per replay cycle):

    runs/<run>/replay_viz/ep{N}/
        sys_00/
            x0.pdb           single-frame PDB (prior placement)
            x1_flow.pdb      single-frame PDB (flow model prediction)
            x1_relaxed.pdb   single-frame PDB (final UMA-relaxed, if converged)
            traj.xyz         multi-frame extended XYZ (UMA relax trajectory)
            data.npz         per-step arrays: positions, energy, fmax
            meta.json        static metadata (sid, E_gt, E_pred_final, tags, ...)
        sys_01/
            ...
        _index.json          top-level list of captured systems + global metadata

Each replay cycle REPLACES the previous cycle's viz dir (no accumulation).
"""
from __future__ import annotations

import json
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


class FixedAtomsHook:
    """Zero forces and velocities on fixed atoms each FIRE step.

    Equivalent to ASE's ``FixAtoms`` constraint inside a batched dynamics
    integrator (nvalchemi's ``FIRE`` doesn't ship a fixed-atom constraint
    natively, so we mask in a hook). Fires at ``AFTER_COMPUTE`` so forces are
    zeroed *before* the integrator's ``post_update`` step uses them.

    Args
    ----
    fixed_mask : (N_total,) bool tensor — True for atoms to freeze, in the
        same flat node ordering as the Batch.
    """

    def __init__(self, fixed_mask):
        import torch as _torch
        from nvalchemi.dynamics import DynamicsStage  # lazy import
        self.fixed_mask = fixed_mask if isinstance(fixed_mask, _torch.Tensor) \
            else _torch.as_tensor(fixed_mask, dtype=_torch.bool)
        self.frequency: int = 1
        self.stage = DynamicsStage.AFTER_COMPUTE

    def __call__(self, ctx, stage) -> None:  # noqa: ARG002
        b = ctx.batch
        m = self.fixed_mask
        if getattr(b, "forces", None) is not None:
            if m.device != b.forces.device:
                self.fixed_mask = m = m.to(b.forces.device)
            b.forces[m] = 0.0
        if getattr(b, "velocities", None) is not None:
            if m.device != b.velocities.device:
                self.fixed_mask = m = m.to(b.velocities.device)
            b.velocities[m] = 0.0


class TrajectoryHook:
    """nvalchemi-compatible Hook that captures per-step positions/energy/fmax
    for a specified subset of systems in the current FIRE batch.

    Register via ``fire.register_hook(hook)`` or pass to ``FIRE(..., hooks=[hook])``.
    Fires at DynamicsStage.AFTER_STEP every step (frequency=1).

    Attributes
    ----------
    target_local_indices : list of int
        Indices within the FIRE batch to capture. Other systems are ignored
        (no overhead).
    trajectories : dict of {local_idx: {"positions": [...], "energy": [...], "fmax": [...]}}
        Filled after each FIRE step. positions are (N_atoms, 3) numpy float32 arrays.
    """

    def __init__(self, target_local_indices: Sequence[int]):
        from nvalchemi.dynamics import DynamicsStage  # lazy import
        self.target_local_indices: List[int] = list(target_local_indices)
        self.frequency: int = 1
        self.stage: Enum = DynamicsStage.AFTER_STEP
        self.trajectories: Dict[int, Dict[str, List[Any]]] = {
            i: {"positions": [], "energy": [], "fmax": []}
            for i in self.target_local_indices
        }

    def __call__(self, ctx, stage) -> None:  # noqa: ARG002
        batch = ctx.batch
        ptr = batch.batch_ptr.long().tolist()
        energies = batch.energy.squeeze(-1).detach()
        forces = batch.forces.detach()
        positions = batch.positions.detach()

        for i in self.target_local_indices:
            start, end = ptr[i], ptr[i + 1]
            pos = positions[start:end].cpu().numpy().astype(np.float32).copy()
            e = float(energies[i].item())
            fmax = float(forces[start:end].norm(dim=-1).max().item())
            self.trajectories[i]["positions"].append(pos)
            self.trajectories[i]["energy"].append(e)
            self.trajectories[i]["fmax"].append(fmax)


def pick_viz_indices(n_total: int, n: int, seed: int) -> List[int]:
    """Deterministically select up to ``n`` indices from ``range(n_total)``."""
    if n_total <= n:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_total, size=n, replace=False).tolist())


def rotate_viz_dir(viz_root: Path, current_epoch: int) -> None:
    """Delete every ``ep*`` subdirectory except the one for ``current_epoch``."""
    if not viz_root.exists():
        return
    keep = f"ep{current_epoch}"
    for child in viz_root.iterdir():
        if child.is_dir() and child.name.startswith("ep") and child.name != keep:
            shutil.rmtree(child, ignore_errors=True)


def _atoms(numbers, positions, cell, tags):
    from ase import Atoms
    return Atoms(
        numbers=np.asarray(numbers).astype(np.int64),
        positions=np.asarray(positions).astype(np.float64),
        cell=np.asarray(cell).astype(np.float64),
        pbc=True,
        tags=np.asarray(tags).astype(np.int64).tolist(),
    )


def reference_center_translation(positions, cell) -> np.ndarray:
    """Translation that ``ase.Atoms.center()`` would apply to ``positions``.

    Captured as a vector so multi-frame trajectories can be translated by the
    SAME offset (no drift across frames). ``Atoms.center()`` itself moves the
    centroid to ``0.5*(a+b+c)`` in Cartesian space.
    """
    a = _atoms([1] * len(positions), positions, cell, tags=[0] * len(positions))
    pre = a.get_positions().mean(axis=0)
    a.center()
    return (a.get_positions().mean(axis=0) - pre).astype(np.float64)


def save_structure_pdb(numbers, positions, cell, tags, path: Path, offset=None) -> None:
    """Single-frame PDB with CRYST1 cell record. Readable by NGL.js.

    If ``offset`` is None, calls ``atoms.center()`` (centroid → cell center).
    Pass an explicit offset to keep multi-snapshot views (x_0, x_1_flow,
    x_1_relaxed) aligned across files.
    """
    from ase.io import write as ase_write
    atoms = _atoms(numbers, positions, cell, tags)
    if offset is None:
        atoms.center()
    else:
        atoms.translate(np.asarray(offset, dtype=np.float64))
    ase_write(str(path), atoms, format="proteindatabank")


def _save_trajectory(numbers, traj_positions, cell, tags, path: Path,
                     offset, ase_format: str) -> None:
    from ase.io import write as ase_write
    frames = [_atoms(numbers, p, cell, tags) for p in traj_positions]
    if offset is None:
        # No external offset → derive from the first frame so the rest of
        # the trajectory is aligned to that frame's centering.
        pre = frames[0].get_positions().mean(axis=0)
        frames[0].center()
        offset = frames[0].get_positions().mean(axis=0) - pre
        for fr in frames[1:]:
            fr.translate(offset)
    else:
        offset = np.asarray(offset, dtype=np.float64)
        for fr in frames:
            fr.translate(offset)
    ase_write(str(path), frames, format=ase_format)


def save_trajectory_xyz(numbers, traj_positions, cell, tags, path: Path, offset=None) -> None:
    """Multi-frame extended XYZ (for ASE/OVITO). NGL doesn't read xyz."""
    _save_trajectory(numbers, traj_positions, cell, tags, path, offset, "extxyz")


def save_trajectory_pdb(numbers, traj_positions, cell, tags, path: Path, offset=None) -> None:
    """Multi-model PDB (one MODEL/ENDMDL per frame). NGL native."""
    _save_trajectory(numbers, traj_positions, cell, tags, path, offset, "proteindatabank")


def save_traj_npz(traj_data: Dict[str, List[Any]], path: Path) -> None:
    """Per-step positions, energy, fmax as a compressed npz."""
    np.savez_compressed(
        str(path),
        positions=np.stack(traj_data["positions"], axis=0).astype(np.float32),
        energy=np.asarray(traj_data["energy"], dtype=np.float32),
        fmax=np.asarray(traj_data["fmax"], dtype=np.float32),
    )


def save_meta_json(meta: Dict[str, Any], path: Path) -> None:
    # Convert any numpy scalars / arrays to native Python for json compatibility.
    def _cast(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return [_cast(x) for x in v]
        if isinstance(v, dict):
            return {k: _cast(x) for k, x in v.items()}
        return v

    with open(path, "w") as f:
        json.dump(_cast(meta), f, indent=2)


def write_index(viz_ep_dir: Path, entries: List[Dict[str, Any]]) -> None:
    """Top-level _index.json listing all captured systems and their status."""
    payload = {
        "epoch_dir": viz_ep_dir.name,
        "n_systems": len(entries),
        "systems": entries,
    }
    save_meta_json(payload, viz_ep_dir / "_index.json")
