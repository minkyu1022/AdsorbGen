"""MLIP energy functions for FK steering.

Provides a batched energy callable with the signature

    energy_fn(x_pred, pad_mask, movable_mask) -> (B,)

that ``adsorbgen.flow.euler_sample`` expects. The batch context (atomic
numbers, cell) that the energy model needs on top of ``x_pred`` is captured
through a thin wrapper built by ``make_fk_energy_fn`` so that the flow
sampler stays agnostic about which MLIP is backing the steering.

UMAEnergy loads a fairchem-core pretrained predict unit (default
``uma-s-1p1``) with ``task_name="oc20"`` and returns per-atom-normalized
energies. UMA's graph builder asserts that ``pbc`` is uniformly True or
False, so we feed it ``pbc=True`` — the OC20 slab cells already contain
~10+ Å of vacuum in z, which is effectively non-periodic along that axis.

Reference: AtomMOF's ``src/models/energy/mlip.py`` (per-atom normalization,
batch collation via ``data_list_collater``).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from ase import Atoms


class UMAEnergy(nn.Module):
    """Per-atom-normalized UMA energy over a padded batch of slab+adsorbate systems.

    Args:
        model_name: fairchem pretrained model id (e.g. ``"uma-s-1p1"``).
        device: ``"cuda"`` or ``"cpu"``. Defaults to ``"cuda"`` if available.
        task_name: fairchem task name; ``"oc20"`` for catalysis systems.
        normalize_per_atom: if True (default), divide returned energies by
            the number of real atoms in each system so that batch systems
            with different N are comparable as an FK potential.
    """

    def __init__(
        self,
        model_name: str = "uma-s-1p1",
        device: str | None = None,
        task_name: str = "oc20",
        normalize_per_atom: bool = True,
    ):
        super().__init__()
        # Imported lazily so the rest of adsorbgen still imports cleanly on
        # machines without fairchem-core installed.
        from fairchem.core import pretrained_mlip  # noqa: F401

        self.model_name = model_name
        self.task_name = task_name
        self.normalize_per_atom = normalize_per_atom
        device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(device_str).startswith("cuda"):
            device_str = "cuda"
        self._device = str(device_str)
        self.predictor = pretrained_mlip.get_predict_unit(
            model_name, device=self._device
        )

    @torch.no_grad()
    def forward(
        self,
        cart_coords: torch.Tensor,  # (B, N, 3)
        cell: torch.Tensor,          # (B, 3, 3)
        atomic_numbers: torch.Tensor,  # (B, N) long
        pad_mask: torch.Tensor,      # (B, N) bool
    ) -> torch.Tensor:
        from fairchem.core.datasets import data_list_collater
        from fairchem.core.datasets.atomic_data import AtomicData

        B = cart_coords.shape[0]
        coords_np = cart_coords.detach().cpu().numpy()
        cell_np = cell.detach().cpu().numpy()
        nums_np = atomic_numbers.detach().cpu().numpy()
        pad_np = pad_mask.detach().cpu().bool().numpy()

        data_list = []
        for i in range(B):
            m = pad_np[i]
            atoms = Atoms(
                numbers=nums_np[i][m],
                positions=coords_np[i][m],
                cell=cell_np[i],
                pbc=True,  # UMA requires uniform pbc; slab vacuum makes z effectively non-periodic
            )
            data_list.append(
                AtomicData.from_ase(
                    atoms,
                    task_name=self.task_name,
                    r_edges=False,
                    r_data_keys=["spin", "charge"],
                )
            )
        batch = data_list_collater(data_list, otf_graph=True)
        out = self.predictor.predict(batch)
        energy = out["energy"].detach().to(cart_coords.device, dtype=cart_coords.dtype)

        if self.normalize_per_atom:
            n_atoms = pad_mask.sum(dim=1).to(energy.dtype).clamp_min(1.0)
            energy = energy / n_atoms
        return energy


class UMAForce(nn.Module):
    """UMA single-point forces over a padded batch.

    The force model is used as a detached Langevin-parametrization input. It
    follows the same padded-batch to fairchem AtomicData conversion as
    :class:`UMAEnergy`, but scatters the concatenated force output back to
    ``(B, N, 3)`` with zeros on padding.
    """

    def __init__(
        self,
        model_name: str = "uma-s-1p2",
        device: str | None = None,
        task_name: str = "oc20",
    ):
        super().__init__()
        from fairchem.core import pretrained_mlip  # noqa: F401

        self.model_name = model_name
        self.task_name = task_name
        device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fairchem's pretrained_mlip API accepts only "cuda" or "cpu".
        # Lightning/worker code may pass torch device strings such as
        # "cuda:3"; the current CUDA device has already been selected there.
        if str(device_str).startswith("cuda"):
            device_str = "cuda"
        self._device = str(device_str)
        self.predictor = pretrained_mlip.get_predict_unit(
            model_name, device=self._device,
        )

    @torch.no_grad()
    def forward(
        self,
        cart_coords: torch.Tensor,
        cell: torch.Tensor,
        atomic_numbers: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        from fairchem.core.datasets import data_list_collater
        from fairchem.core.datasets.atomic_data import AtomicData

        B, N, _ = cart_coords.shape
        coords_np = cart_coords.detach().cpu().numpy()
        cell_np = cell.detach().cpu().numpy()
        nums_np = atomic_numbers.detach().cpu().numpy()
        pad_cpu = pad_mask.detach().cpu().bool()
        pad_np = pad_cpu.numpy()

        data_list = []
        counts = []
        for i in range(B):
            m = pad_np[i]
            counts.append(int(m.sum()))
            atoms = Atoms(
                numbers=nums_np[i][m],
                positions=coords_np[i][m],
                cell=cell_np[i],
                pbc=True,
            )
            data_list.append(
                AtomicData.from_ase(
                    atoms,
                    task_name=self.task_name,
                    r_edges=False,
                    r_data_keys=["spin", "charge"],
                )
            )

        batch = data_list_collater(data_list, otf_graph=True)
        out = self.predictor.predict(batch)
        forces_cat = out["forces"].detach().to(
            cart_coords.device, dtype=cart_coords.dtype,
        )
        forces = cart_coords.new_zeros((B, N, 3))
        start = 0
        for i, n in enumerate(counts):
            end = start + n
            if n:
                idx = pad_cpu[i].to(device=cart_coords.device)
                forces[i, idx] = forces_cat[start:end]
            start = end
        return forces


class UMARelaxer:
    """LBFGS relaxation via fairchem FAIRChemCalculator.

    Reuses the exact logic of scripts/phase3_adsorption.py. Given a list of
    ASE Atoms (init poses), relaxes each and returns E_sys + metadata.
    """

    def __init__(
        self,
        model_name: str = "uma-s-1p1",
        task_name: str = "oc20",
        device: str | None = None,
        fmax: float = 0.05,
        max_steps: int = 100,
    ):
        from fairchem.core import pretrained_mlip
        self.model_name = model_name
        self.task_name = task_name
        self.fmax = float(fmax)
        self.max_steps = int(max_steps)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predict_unit = pretrained_mlip.get_predict_unit(
            model_name, device=self._device,
        )

    @torch.no_grad()
    def relax_one(self, atoms):
        """Relax one Atoms (copied); return dict with keys:
        atoms, E_sys, forces_max, converged, n_steps, error."""
        from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
        from ase.optimize import LBFGS
        import numpy as np

        atoms = atoms.copy()
        atoms.calc = FAIRChemCalculator(self.predict_unit, task_name=self.task_name)
        opt = LBFGS(atoms, logfile=None)
        try:
            converged = opt.run(fmax=self.fmax, steps=self.max_steps)
            E = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            fmax_val = float(np.max(np.linalg.norm(forces, axis=1)))
            err = None
        except Exception as e:
            converged, E, fmax_val, err = False, float("nan"), float("nan"), str(e)
        return {
            "atoms": atoms, "E_sys": E, "forces_max": fmax_val,
            "converged": bool(converged),
            "n_steps": int(opt.nsteps) if err is None else 0,
            "error": err,
        }

    def relax_batch(self, atoms_list):
        return [self.relax_one(a) for a in atoms_list]


def make_fk_energy_fn(
    energy_model: nn.Module,
    atomic_numbers: torch.Tensor,
    cell: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Bind static batch context so the returned callable matches the
    ``energy_fn(x_pred, pad_mask, movable_mask) -> (B,)`` signature that
    ``flow.euler_sample`` feeds FK steering.

    ``atomic_numbers`` and ``cell`` must already be replicated to match the
    (B*P)-sized steering batch the sampler is iterating over.
    """

    def _fn(
        x_pred: torch.Tensor,
        pad_mask: torch.Tensor,
        movable_mask: torch.Tensor,  # unused (captured for API compat)
    ) -> torch.Tensor:
        del movable_mask
        return energy_model(x_pred, cell, atomic_numbers, pad_mask)

    return _fn
