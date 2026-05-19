#!/usr/bin/env python
"""Compute the flow inference trajectory for one system from a SAVED x0.

Flow Euler sampling is a deterministic ODE: given the exact x0 and model the
trajectory is fully reproducible. This loads x0 from a viz folder's ``x0.pdb``,
feeds it through ``euler_sample(return_trajectory=True)`` and writes
``flow_traj.xyz`` — unlike ``replay_viz_for_sids.py`` which re-samples a fresh
random placement (and so may not reproduce a rare success).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read

_REPO = Path(__file__).resolve().parents[1]
if (_REPO / "adsorbgen").is_dir():
    sys.path.insert(0, str(_REPO))

from adsorbgen.dataset import (  # noqa: E402
    PreprocessedDisplacementDataset, PlacementPriorDataset, collate_displacement,
)
from adsorbgen.flow import FlowConfig, euler_sample  # noqa: E402
from adsorbgen.model import DiTDenoiserConfig  # noqa: E402
from adsorbgen.model_v2 import DiTDenoiserV2Config  # noqa: E402
from adsorbgen.model_factory import build_model  # noqa: E402
from adsorbgen.replay_viz import save_trajectory_xyz, save_trajectory_pdb  # noqa: E402


def load_model(ckpt_path: Path, device):
    torch.serialization.add_safe_globals(
        [DiTDenoiserConfig, DiTDenoiserV2Config, FlowConfig])
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    model = build_model(hp["model_cfg"])
    sd = ck["state_dict"]
    model.load_state_dict(
        {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")},
        strict=False,
    )
    model.to(device).eval()
    return model, hp["flow_cfg"]


def _model_cfg(m):
    while hasattr(m, "module"):
        m = m.module
    return getattr(m, "cfg", None)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--train-lmdb", required=True)
    p.add_argument("--sid", type=int, required=True)
    p.add_argument("--x0-pdb", required=True, help="saved x0.pdb fed as the start placement")
    p.add_argument("--out-dir", required=True, help="dir to write flow_traj.xyz/.pdb")
    p.add_argument("--prior-mode", default="random_heuristic")
    p.add_argument("--flow-steps", type=int, default=50)
    p.add_argument("--verify-x1-flow", default="", help="optional saved x1_flow.pdb to compare")
    args = p.parse_args()

    device = torch.device("cuda")
    model, flow_cfg = load_model(Path(args.ckpt), device)
    use_ads_ref = bool(getattr(_model_cfg(model), "use_ads_ref_pos", False))
    print(f"use_ads_ref_pos = {use_ads_ref}")

    base = PreprocessedDisplacementDataset(args.train_lmdb, max_samples=None)
    idx = None
    for i in range(len(base)):
        if int(base[i]["sid"].item()) == args.sid:
            idx = i
            break
    if idx is None:
        raise SystemExit(f"sid {args.sid} not found in {args.train_lmdb}")
    print(f"sid {args.sid} -> dataset idx {idx}")

    placement_ds = PlacementPriorDataset(
        args.train_lmdb, prior_mode=args.prior_mode,
        max_samples=None, provide_ads_ref_pos=use_ads_ref,
    )
    batch = collate_displacement([placement_ds[idx]])
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}

    n = int(batch["pad_mask"][0].sum().item())
    x0_atoms = read(args.x0_pdb)
    x0 = np.asarray(x0_atoms.get_positions(), dtype=np.float32)
    z_pdb = np.asarray(x0_atoms.get_atomic_numbers(), dtype=np.int64)
    z_sys = batch["atomic_numbers"][0, :n].cpu().numpy().astype(np.int64)
    if x0.shape[0] != n:
        raise SystemExit(f"x0.pdb atom count {x0.shape[0]} != system {n}")
    if not np.array_equal(z_pdb, z_sys):
        raise SystemExit("atom order mismatch between x0.pdb and system input")

    # Inject the saved x0 (overrides the random placement for both the euler
    # integrator's pos_0 and the model's pos reference).
    batch["pos"][0, :n, :] = torch.as_tensor(
        x0, dtype=batch["pos"].dtype, device=device)

    def fwd(x_t, t, _b=batch):
        extra = {"ads_ref_pos": _b["ads_ref_pos"]} if use_ads_ref else {}
        return model(
            pos=_b["pos"], x_t=x_t, t=t,
            atomic_numbers=_b["atomic_numbers"], tags=_b["tags"],
            movable_mask=_b["movable_mask"], pad_mask=_b["pad_mask"],
            cell=_b["cell"], **extra,
        )

    es = euler_sample(
        fwd, batch["pos"], batch["movable_mask"], batch["pad_mask"],
        flow_cfg, num_steps=args.flow_steps, return_trajectory=True,
    )
    x_out = es["x_out"]
    traj = es["x_trajectory"][:, 0, :n, :].cpu().numpy().astype(np.float32)

    tags = batch["tags"][0, :n].cpu().numpy()
    cell = batch["cell"][0].cpu().numpy()
    if cell.ndim == 3:
        cell = cell[0]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_trajectory_xyz(z_sys, traj, cell, tags, out / "flow_traj.xyz")
    save_trajectory_pdb(z_sys, traj, cell, tags, out / "flow_traj.pdb")
    print(f"wrote flow_traj ({traj.shape[0]} frames) -> {out}")

    if args.verify_x1_flow:
        ref = np.asarray(read(args.verify_x1_flow).get_positions(), dtype=np.float64)
        got = x_out[0, :n, :].cpu().numpy().astype(np.float64)
        rmsd = float(np.sqrt(((ref - got) ** 2).sum(1).mean()))
        verdict = "MATCH" if rmsd < 0.05 else "MISMATCH"
        print(f"VERIFY x1_flow vs saved: RMSD = {rmsd:.5f} A  ({verdict})")


if __name__ == "__main__":
    main()
