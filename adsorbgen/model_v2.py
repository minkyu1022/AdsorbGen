"""DiTDenoiserV2: single uniform DiT backbone for AdsorbGen flow matching.

V2 collapses v1's encoder-trunk-decoder hierarchy (which is degenerate when
atom and token counts are 1:1) into a single ``DiT`` stack with
AttentionPairBias. The pair representation is built once from MIC geometry,
enriched by a one-shot outer product, and reused by every block.

V2 is unconditional: there is no ΔE conditioning or classifier-free dropout.
The model only sees (geometry, time).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from adsorbgen.model import CellEmbedder, _pair_diff_mic
from adsorbgen.transformer import DiT, TimestepEmbedder


@dataclass
class DiTDenoiserV2Config:
    dim: int = 512
    pair_dim: int = 128
    depth: int = 13
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    num_elements: int = 100
    num_tags: int = 3
    sigma: float = 0.5
    activation_checkpointing: bool = False


class DiTDenoiserV2(nn.Module):
    """Single uniform DiT denoiser with AttentionPairBias.

    Args:
        cfg: ``DiTDenoiserV2Config`` with the 13 locked fields.

    Inputs (forward):
        pos:            (B, N, 3) Cartesian reference positions
        delta_t:        (B, N, 3) current noisy displacement
        t:              (B,) flow time in [eps, 1-eps]
        atomic_numbers: (B, N) long
        tags:           (B, N) long in {0, 1, 2}
        movable_mask:   (B, N) bool, True = movable adsorbate atom
        pad_mask:       (B, N) bool, True = real atom
        cell:           (B, 3, 3)

    Output:
        (B, N, 3) predicted displacement; zero on non-movable / padded atoms.
    """

    def __init__(self, cfg: DiTDenoiserV2Config):
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim
        pair_dim = cfg.pair_dim

        self.atom_embed = nn.Embedding(cfg.num_elements, dim)
        self.tag_embed = nn.Embedding(cfg.num_tags, dim)
        self.movable_embed = nn.Embedding(2, dim)
        self.pos_proj = nn.Linear(3, dim, bias=True)
        self.xt_proj = nn.Linear(3, dim, bias=True)

        self.emb_pair_pos = nn.Linear(3, pair_dim, bias=False)
        self.emb_pair_dist = nn.Linear(1, pair_dim, bias=False)
        self.emb_pair_mask = nn.Linear(1, pair_dim, bias=False)
        self.emb_pair_ads = nn.Linear(1, pair_dim, bias=False)

        self.pair_outer_a = nn.Linear(dim, pair_dim, bias=False)
        self.pair_outer_b = nn.Linear(dim, pair_dim, bias=False)
        self.pair_norm = nn.LayerNorm(pair_dim)

        self.t_embedder = TimestepEmbedder(hidden_dim=dim)
        self.cell_embedder = CellEmbedder(hidden_dim=dim)

        self.backbone = DiT(
            dim=dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            pair_dim=pair_dim,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            activation_checkpointing=cfg.activation_checkpointing,
        )

        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(dim, 3, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _embed_tokens(
        self,
        pos: torch.Tensor,
        delta_t: torch.Tensor,
        atomic_numbers: torch.Tensor,
        tags: torch.Tensor,
        movable_mask: torch.Tensor,
        pad_mask: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        # tokens: (B, N, dim)
        x_t = pos + delta_t
        cell_emb = self.cell_embedder(cell).unsqueeze(1)  # (B, 1, dim)
        tokens = (
            self.atom_embed(atomic_numbers.clamp_max(self.cfg.num_elements - 1))
            + self.tag_embed(tags.clamp(0, self.cfg.num_tags - 1))
            + self.movable_embed(movable_mask.long())
            + self.pos_proj(pos)
            + self.xt_proj(x_t)
            + cell_emb
        )
        return tokens * pad_mask.unsqueeze(-1).to(tokens.dtype)

    def _build_pair_features(
        self,
        pos: torch.Tensor,
        tags: torch.Tensor,
        pad_mask: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        # pair: (B, N, N, pair_dim)
        diff = _pair_diff_mic(pos, cell)
        dist2 = (diff * diff).sum(dim=-1, keepdim=True)
        dist_feat = 1.0 / (1.0 + dist2)

        non_bulk = (tags >= 1)
        pad_pair = pad_mask.unsqueeze(2) & pad_mask.unsqueeze(1)
        v = (pad_pair & non_bulk.unsqueeze(2) & non_bulk.unsqueeze(1)).to(pos.dtype).unsqueeze(-1)

        ads = (tags == 2)
        ads_pair = (ads.unsqueeze(2) & ads.unsqueeze(1)).to(pos.dtype).unsqueeze(-1)

        pair = (
            self.emb_pair_pos(diff) * v
            + self.emb_pair_dist(dist_feat) * v
            + self.emb_pair_mask(v) * v
            + self.emb_pair_ads(ads_pair) * v
        )
        return pair

    def forward(
        self,
        pos: torch.Tensor,
        delta_t: torch.Tensor,
        t: torch.Tensor,
        atomic_numbers: torch.Tensor,
        tags: torch.Tensor,
        movable_mask: torch.Tensor,
        pad_mask: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.isfinite(pos).all():
            raise RuntimeError("Non-finite values in pos")
        if not torch.isfinite(delta_t).all():
            raise RuntimeError("Non-finite values in delta_t")

        tokens = self._embed_tokens(
            pos=pos, delta_t=delta_t, atomic_numbers=atomic_numbers,
            tags=tags, movable_mask=movable_mask, pad_mask=pad_mask, cell=cell,
        )  # (B, N, dim)

        pair_geom = self._build_pair_features(pos, tags, pad_mask, cell)  # (B, N, N, pair_dim)

        # One-shot outer-product enrichment, then a single LayerNorm.
        a = self.pair_outer_a(tokens)  # (B, N, pair_dim)
        b = self.pair_outer_b(tokens)  # (B, N, pair_dim)
        pair = pair_geom + a.unsqueeze(2) + b.unsqueeze(1)  # (B, N, N, pair_dim)
        pair = self.pair_norm(pair)

        c = self.t_embedder(t)  # (B, dim)

        x = self.backbone(tokens, c, pad_mask, pair)  # (B, N, dim)

        x = self.out_norm(x)
        out = self.out_proj(x)  # (B, N, 3)

        movable_f = movable_mask.unsqueeze(-1).to(out.dtype)
        out = out * movable_f

        if self.training and not torch.isfinite(out).all():
            raise RuntimeError("NaN detected in DiTDenoiserV2 forward")
        return out
