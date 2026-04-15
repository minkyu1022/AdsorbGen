"""DiT transformer blocks with AttentionPairBias (AF3-style).

Provenance:
    AtomMOF/src/models/simple/transformers.py (DiT, DiTBlock, TimestepEmbedder, Mlp)
    AtomMOF/src/models/simple/attention.py    (AttentionPairBias)

Differences vs. original:
    - Dropped xformers / einops dependency (PyTorch SDPA only).
    - TimestepEmbedder accepts continuous t in [0, 1] rather than integer steps;
      we scale to [0, 1000] internally so the sinusoidal frequencies match the
      DiT paper's schedule.
    - Pair bias is a dense ``(B, N, N, pair_dim)`` tensor built by the caller
      (``DiTDenoiser._build_pair_features``) and reused across layers. Each
      ``MaskedSelfAttention`` owns its own ``LayerNorm(pair_dim) + Linear(pair_dim,
      num_heads)`` and adds the projected per-head bias to the attention logits.
    - ``AdaLN_modulation`` last linear is zero-initialized (unchanged from DiT paper).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift, broadcast over token dim."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Continuous timestep embedding."""

    def __init__(self, hidden_dim: int, frequency_embedding_dim: int = 256, time_scale: float = 1000.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim
        self.time_scale = time_scale

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t * self.time_scale, self.frequency_embedding_dim)
        return self.mlp(t_freq)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


class MaskedSelfAttention(nn.Module):
    """Multi-head self-attention with boolean padding mask and optional pair bias.

    When ``pair_dim > 0``, a per-head scalar bias derived from the shared
    ``(B, N, N, pair_dim)`` pair representation is added to the attention logits
    (AlphaFold-3 AttentionPairBias). When ``pair_dim == 0``, standard masked
    self-attention is used (no pair bias).
    """

    def __init__(self, dim: int, num_heads: int, pair_dim: int = 0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pair_dim = pair_dim
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj_g = nn.Linear(dim, dim, bias=False)
        self.proj_o = nn.Linear(dim, dim, bias=False)

        if pair_dim > 0:
            self.pair_norm = nn.LayerNorm(pair_dim)
            self.pair_to_heads = nn.Linear(pair_dim, num_heads, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pair_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        h = self.num_heads
        d = self.head_dim

        s = self.norm(x)
        qkv = self.qkv(s)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, h, d).transpose(1, 2)
        k = k.view(B, N, h, d).transpose(1, 2)
        v = v.view(B, N, h, d).transpose(1, 2)

        key_mask = mask[:, None, None, :]
        attn_bias = torch.zeros(B, 1, 1, N, device=x.device, dtype=q.dtype)
        attn_bias = attn_bias.masked_fill(~key_mask, float("-inf"))

        if self.pair_dim > 0 and pair_feats is not None:
            pair_logits = self.pair_to_heads(self.pair_norm(pair_feats))  # (B, N, N, H)
            pair_logits = pair_logits.permute(0, 3, 1, 2).to(q.dtype)  # (B, H, N, N)
            attn_bias = attn_bias + pair_logits

        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        o = o.transpose(1, 2).reshape(B, N, D)

        g = torch.sigmoid(self.proj_g(s))
        return self.proj_o(g * o)


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning and optional pair bias."""

    def __init__(self, dim: int, num_heads: int, pair_dim: int = 0, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.pair_dim = pair_dim
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MaskedSelfAttention(dim=dim, num_heads=num_heads, pair_dim=pair_dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        if self.pair_dim > 0:
            nn.init.zeros_(self.attn.pair_to_heads.weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor, pair_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask, pair_feats)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiT(nn.Module):
    """Stack of DiTBlocks with optional pair-bias representation."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        pair_dim: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.layers = nn.ModuleList(
            [
                DiTBlock(dim=dim, num_heads=num_heads, pair_dim=pair_dim, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor, pair_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, c, mask, pair_feats, use_reentrant=False)
            else:
                x = layer(x, c, mask, pair_feats)
        return x
