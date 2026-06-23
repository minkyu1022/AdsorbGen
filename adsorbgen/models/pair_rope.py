"""Pair-level 3D RoPE for attention logits."""

from __future__ import annotations

import torch
import torch.nn as nn


class PairRoPE3D(nn.Module):
    """Pair-level 3D RoPE using MIC displacement vectors in Angstrom."""

    def __init__(self, rope_dim: int, head_dim: int, base: float = 10.0):
        super().__init__()
        if rope_dim % 6 != 0:
            raise ValueError("rope_dim must be divisible by 6")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim {rope_dim} > head_dim {head_dim}")
        self.rope_dim = int(rope_dim)
        self.dim_per_axis = self.rope_dim // 3
        n_freqs = self.dim_per_axis // 2
        freqs = 1.0 / (
            float(base) ** (torch.arange(0, 2 * n_freqs, 2, dtype=torch.float32) / (2 * n_freqs))
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def precompute(self, diff_mic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_parts, sin_parts = [], []
        for axis in range(3):
            angles = diff_mic[..., axis:axis + 1] * self.freqs.to(device=diff_mic.device)
            cos_parts.append(angles.cos().repeat(1, 1, 1, 2))
            sin_parts.append(angles.sin().repeat(1, 1, 1, 2))
        cos = torch.cat(cos_parts, dim=-1)
        sin = torch.cat(sin_parts, dim=-1)
        return cos.to(diff_mic.dtype), sin.to(diff_mic.dtype)

    def score(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Return unscaled q_i^T R(diff_ij) k_j over the RoPE channels."""
        rd = self.rope_dim
        q_r = q[..., :rd].contiguous()
        k_r = k[..., :rd].contiguous()
        cos = cos.to(q_r.dtype)
        sin = sin.to(q_r.dtype)
        x1, x2 = k_r.chunk(2, dim=-1)
        k_half = torch.cat((-x2, x1), dim=-1)
        return (
            torch.einsum("bhir,bhjr,bijr->bhij", q_r, k_r, cos)
            + torch.einsum("bhir,bhjr,bijr->bhij", q_r, k_half, sin)
        )
