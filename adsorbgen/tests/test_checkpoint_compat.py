"""Checkpoint-loading tests for the current AdsorbGen config format."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from adsorbgen.inference import _extract_state_dict, _resolve_model_cfg
from adsorbgen.models.dit import DiTDenoiser, DiTDenoiserConfig
from adsorbgen.models.factory import build_model
from adsorbgen.training.train_cli import _check_resume_arch


def _tiny_cfg() -> DiTDenoiserConfig:
    return DiTDenoiserConfig(
        atom_s=32,
        atom_z=16,
        token_s=32,
        token_z=16,
        enc_depth=1,
        trunk_depth=1,
        dec_depth=1,
        enc_heads=2,
        trunk_heads=2,
        dec_heads=2,
        mlp_ratio=2.0,
        num_elements=80,
        use_ads_ref_pos=True,
        use_ads_center_rel_head=True,
    )


def _args_payload(cfg: DiTDenoiserConfig) -> dict:
    return {
        "arch": "v1",
        "model_config": asdict(cfg),
        "flow_config": {"eps": 1e-5, "prediction_type": "x1"},
    }


def _batch(n_atoms: int = 6):
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(1, n_atoms, 3, generator=g)
    x_t = pos + 0.1 * torch.randn(1, n_atoms, 3, generator=g)
    atomic_numbers = torch.randint(1, 50, (1, n_atoms), generator=g)
    tags = torch.tensor([[0, 1, 1, 2, 2, 2]])[:, :n_atoms]
    movable_mask = tags > 0
    pad_mask = torch.ones(1, n_atoms, dtype=torch.bool)
    cell = torch.eye(3).unsqueeze(0) * 10.0
    return {
        "pos": pos,
        "x_t": x_t,
        "t": torch.tensor([0.5]),
        "atomic_numbers": atomic_numbers,
        "tags": tags,
        "movable_mask": movable_mask,
        "pad_mask": pad_mask,
        "cell": cell,
        "ads_ref_pos": torch.zeros_like(pos),
    }


def test_current_args_json_round_trip(tmp_path: Path):
    cfg = _tiny_cfg()
    args_json = tmp_path / "args.json"
    args_json.write_text(json.dumps(_args_payload(cfg)))

    rebuilt_cfg = _resolve_model_cfg(args_json)
    assert isinstance(rebuilt_cfg, DiTDenoiserConfig)
    assert rebuilt_cfg.num_elements == 80
    assert rebuilt_cfg.use_ads_ref_pos is True
    assert rebuilt_cfg.use_ads_center_rel_head is True


def test_current_raw_state_dict_loads(tmp_path: Path):
    cfg = _tiny_cfg()
    args_json = tmp_path / "args.json"
    args_json.write_text(json.dumps(_args_payload(cfg)))

    model = DiTDenoiser(cfg).eval()
    raw_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), raw_path)

    state = torch.load(raw_path, map_location="cpu", weights_only=False)
    sd = _extract_state_dict(state)

    rebuilt = build_model(_resolve_model_cfg(args_json)).eval()
    missing, unexpected = rebuilt.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    out = rebuilt(**_batch())
    assert torch.isfinite(out).all()


def test_current_lightning_state_dict_loads(tmp_path: Path):
    cfg = _tiny_cfg()
    args_json = tmp_path / "args.json"
    args_json.write_text(json.dumps(_args_payload(cfg)))

    model = DiTDenoiser(cfg).eval()
    ckpt = {
        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
        "epoch": 0,
        "global_step": 0,
    }
    ckpt_path = tmp_path / "last.ckpt"
    torch.save(ckpt, ckpt_path)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _extract_state_dict(state)
    assert sd
    assert all(not k.startswith("model.") for k in sd)

    rebuilt = build_model(_resolve_model_cfg(args_json)).eval()
    missing, unexpected = rebuilt.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    out = rebuilt(**_batch())
    assert torch.isfinite(out).all()


def test_resume_arch_check_uses_current_args_schema(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    _check_resume_arch(out_dir, "v1")

    (out_dir / "args.json").write_text(json.dumps(_args_payload(_tiny_cfg())))
    _check_resume_arch(out_dir, "v1")
    with pytest.raises(RuntimeError, match="arch mismatch"):
        _check_resume_arch(out_dir, "v2")
