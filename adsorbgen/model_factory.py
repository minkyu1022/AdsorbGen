"""Single-entry factory for AdsorbGen denoiser models.

Dispatches on the config dataclass type so callers (``train.py``,
``inference.py``) stay arch-agnostic.
"""

from __future__ import annotations

import torch.nn as nn

from adsorbgen.model import DiTDenoiser, DiTDenoiserConfig
from adsorbgen.model_v2 import DiTDenoiserV2, DiTDenoiserV2Config


def build_model(model_cfg) -> nn.Module:
    if isinstance(model_cfg, DiTDenoiserConfig):
        return DiTDenoiser(model_cfg)
    if isinstance(model_cfg, DiTDenoiserV2Config):
        return DiTDenoiserV2(model_cfg)
    raise TypeError(f"Unknown model config type: {type(model_cfg).__name__}")
