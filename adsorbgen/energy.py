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
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
