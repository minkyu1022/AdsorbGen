"""Regression tests for v1↔v2 checkpoint compatibility and resume safety.

Covers:
    1. v1 raw state_dict loads via _resolve_model_cfg + _extract_state_dict.
    2. v1 Lightning ``last.ckpt`` (state["state_dict"] with "model." prefix) loads.
       This regresses a current-main bug where inference.py only handled
       state["model"] / raw dicts.
    3. Non-default v1 fields (delta_e_freq_dim, num_elements) survive a
       save/load roundtrip — regresses the silent-default bug.
    4. _check_resume_arch fails fast when the requested --arch differs from
       the existing run, and is a no-op when they match.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from adsorbgen.inference import _extract_state_dict, _resolve_model_cfg  # noqa: E402
from adsorbgen.model import DiTDenoiser, DiTDenoiserConfig  # noqa: E402
from adsorbgen.model_factory import build_model  # noqa: E402
from adsorbgen.model_v2 import DiTDenoiserV2, DiTDenoiserV2Config  # noqa: E402
from adsorbgen.train import _check_resume_arch  # noqa: E402


def _tiny_v1_cfg() -> DiTDenoiserConfig:
    return DiTDenoiserConfig(
        atom_s=32, atom_z=16, token_s=32, token_z=16,
        enc_depth=1, trunk_depth=1, dec_depth=1,
        enc_heads=2, trunk_heads=2, dec_heads=2,
        mlp_ratio=2.0,
    )


def _tiny_v2_cfg() -> DiTDenoiserV2Config:
    return DiTDenoiserV2Config(
        dim=32, pair_dim=16, depth=2, num_heads=4, mlp_ratio=2.0,
    )


def _make_batch(model_n: int = 6):
    g = torch.Generator().manual_seed(0)
    B, N = 1, model_n
    pos = torch.randn(B, N, 3, generator=g)
    delta_t = torch.randn(B, N, 3, generator=g) * 0.1
    t = torch.tensor([0.5])
    atomic_numbers = torch.randint(1, 50, (B, N), generator=g)
    tags = torch.randint(0, 3, (B, N), generator=g)
    movable_mask = torch.zeros(B, N, dtype=torch.bool)
    movable_mask[:, N // 2:] = True
    pad_mask = torch.ones(B, N, dtype=torch.bool)
    cell = torch.eye(3).unsqueeze(0) * 10.0
    return dict(
        pos=pos, delta_t=delta_t, t=t,
        atomic_numbers=atomic_numbers, tags=tags,
        movable_mask=movable_mask, pad_mask=pad_mask,
        cell=cell, delta_e=torch.zeros(B), cond_drop=torch.zeros(B, dtype=torch.bool),
    )


def _v1_args_payload(cfg: DiTDenoiserConfig) -> dict:
    """Mimic the legacy flat schema (no ``arch`` field)."""
    from dataclasses import asdict
    return asdict(cfg)


def test_v1_raw_state_dict_load(tmp_path: Path):
    cfg = _tiny_v1_cfg()
    args_json = tmp_path / "args.json"
    args_json.write_text(json.dumps(_v1_args_payload(cfg)))

    model = DiTDenoiser(cfg).eval()
    sd_path = tmp_path / "ckpt_last.pt"
    torch.save(model.state_dict(), sd_path)

    state = torch.load(sd_path, map_location="cpu", weights_only=False)
    sd = _extract_state_dict(state)

    rebuilt_cfg = _resolve_model_cfg(args_json)
    assert isinstance(rebuilt_cfg, DiTDenoiserConfig)
    rebuilt = build_model(rebuilt_cfg).eval()
    missing, unexpected = rebuilt.load_state_dict(sd, strict=True)
    assert not missing and not unexpected

    out = rebuilt(**_make_batch())
    assert torch.isfinite(out).all()


def test_v1_lightning_last_ckpt_load(tmp_path: Path):
    """Regression: inference.py used to only handle state['model'] / raw dicts."""
    cfg = _tiny_v1_cfg()
    args_json = tmp_path / "args.json"
    args_json.write_text(json.dumps(_v1_args_payload(cfg)))

    model = DiTDenoiser(cfg).eval()
    lightning_sd = {f"model.{k}": v for k, v in model.state_dict().items()}
    ckpt = {
        "state_dict": lightning_sd,
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": "2.0.0",
    }
    ckpt_path = tmp_path / "last.ckpt"
    torch.save(ckpt, ckpt_path)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _extract_state_dict(state)
    assert all(not k.startswith("model.") for k in sd.keys())

    rebuilt = build_model(_resolve_model_cfg(args_json)).eval()
    missing, unexpected = rebuilt.load_state_dict(sd, strict=True)
    assert not missing and not unexpected

    out = rebuilt(**_make_batch())
    assert torch.isfinite(out).all()


def test_non_default_fields_round_trip(tmp_path: Path):
    """Regression: delta_e_freq_dim / num_elements used to silently fall back to defaults."""
    cfg = DiTDenoiserConfig(
        atom_s=32, atom_z=16, token_s=32, token_z=16,
        enc_depth=1, trunk_depth=1, dec_depth=1,
        enc_heads=2, trunk_heads=2, dec_heads=2,
        mlp_ratio=2.0,
        delta_e_freq_dim=128,
        num_elements=95,
    )
    args_json = tmp_path / "args.json"
    args_json.write_text(json.dumps(_v1_args_payload(cfg)))

    rebuilt = _resolve_model_cfg(args_json)
    assert rebuilt.delta_e_freq_dim == 128
    assert rebuilt.num_elements == 95

    model = build_model(rebuilt)
    assert model.delta_e_embedder.frequency_embedding_dim == 128
    assert model.atom_embed.num_embeddings == 95


def test_resume_arch_check(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    # Empty dir: no-op
    _check_resume_arch(out_dir, "v1")
    _check_resume_arch(out_dir, "v2")

    # v1 args.json
    (out_dir / "args.json").write_text(json.dumps(_v1_args_payload(_tiny_v1_cfg())))
    _check_resume_arch(out_dir, "v1")  # match -> ok
    with pytest.raises(RuntimeError, match="arch mismatch"):
        _check_resume_arch(out_dir, "v2")

    # v2 args.json
    out_dir2 = tmp_path / "run2"
    out_dir2.mkdir()
    (out_dir2 / "args.json").write_text(json.dumps({
        "arch": "v2",
        "model_config": {"dim": 32, "pair_dim": 16, "depth": 2, "num_heads": 4},
    }))
    _check_resume_arch(out_dir2, "v2")  # match -> ok
    with pytest.raises(RuntimeError, match="arch mismatch"):
        _check_resume_arch(out_dir2, "v1")
