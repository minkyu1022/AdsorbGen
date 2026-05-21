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

from adsorbgen.models.dit import CellEmbedder, _pair_diff_mic
from adsorbgen.models.transformer import DiT, DiTCrossAttn, TimestepEmbedder


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
    activation_checkpointing: bool = False

    # ---- architecture-search feature flags ---------------------------------
    # All default to the current v2 baseline so existing ckpts load unchanged.
    use_mic_distance: bool = True       # emb_pair_dist channel (1/(1+d^2) kernel)
    use_pair_position: bool = True      # emb_pair_pos channel (MIC diff vector)
    use_ads_pair: bool = True           # emb_pair_ads channel (ads-ads mask)
    use_ads_surf_pair: bool = False     # emb_pair_ads_surf channel (symmetric ads-surf mask)
    use_pair_outer: bool = True         # outer-product token enrichment
    use_cell_embed: bool = True         # add cell embedding to token features
    use_tag_embed: bool = True          # add tag embedding to token features
    use_movable_embed: bool = True      # add movable-mask embedding to tokens
    # "non_bulk" masks pair features to tag>=1 ⊗ tag>=1; "all" keeps every real pair.
    pair_scope: str = "non_bulk"
    # "reciprocal" = 1/(1+d^2); "gaussian" = sum_k exp(-(d-mu_k)^2/sigma^2) rbf.
    dist_kernel: str = "reciprocal"
    dist_rbf_num: int = 16              # # of gaussian centers if dist_kernel=="gaussian"
    dist_rbf_cutoff: float = 6.0        # max center distance (Å)
    # "3d" wraps all three lattice axes (full periodic); "2d" wraps only a,b
    # (slab geometry — vacuum along c means z should not wrap).
    pair_pbc: str = "3d"
    # When True, pair distances are computed from x_t = pos + delta_t (the noisy
    # intermediate) instead of the static initial pos.  This lets the pair
    # representation track the current denoising state.
    use_dynamic_pair_dist: bool = False

    # Self-conditioning (Chen+22 "Analog Bits", AlphaFold-3): the model takes
    # its own previous delta_1 prediction as an extra input. Training uses a
    # 2-pass trick with 50% probability (see train.py). Inference feeds the
    # previous Euler step's prediction. Backward-compat: default False.
    use_self_cond: bool = False

    # Cross-attention two-stream: replace the single DiT stack with blocks that
    # do (ads self-attn) -> (ads <- surf cross-attn) -> (ads FFN). Surface
    # tokens are a static context; only adsorbate tokens are updated.  Blocks
    # have ~1.5x parameters of a standard DiT block, so depth may be tuned.
    use_cross_attn: bool = False


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
        self.tag_embed = nn.Embedding(cfg.num_tags, dim) if cfg.use_tag_embed else None
        self.movable_embed = nn.Embedding(2, dim) if cfg.use_movable_embed else None
        self.pos_proj = nn.Linear(3, dim, bias=True)
        self.xt_proj = nn.Linear(3, dim, bias=True)

        # Self-conditioning input projection. Zero-init so the variant starts
        # as the no-self-cond baseline and must actively learn to use the
        # previous prediction.
        if cfg.use_self_cond:
            self.prev_pred_proj = nn.Linear(3, dim, bias=False)
            nn.init.zeros_(self.prev_pred_proj.weight)
        else:
            self.prev_pred_proj = None

        self.emb_pair_pos = nn.Linear(3, pair_dim, bias=False) if cfg.use_pair_position else None
        self.emb_pair_mask = nn.Linear(1, pair_dim, bias=False)
        self.emb_pair_ads = nn.Linear(1, pair_dim, bias=False) if cfg.use_ads_pair else None
        self.emb_pair_ads_surf = nn.Linear(1, pair_dim, bias=False) if cfg.use_ads_surf_pair else None

        if cfg.use_mic_distance:
            if cfg.dist_kernel == "reciprocal":
                self.emb_pair_dist = nn.Linear(1, pair_dim, bias=False)
                self.register_buffer("_dist_rbf_centers", torch.empty(0), persistent=False)
            elif cfg.dist_kernel == "gaussian":
                centers = torch.linspace(0.0, cfg.dist_rbf_cutoff, cfg.dist_rbf_num)
                self.register_buffer("_dist_rbf_centers", centers, persistent=False)
                self.emb_pair_dist = nn.Linear(cfg.dist_rbf_num, pair_dim, bias=False)
            else:
                raise ValueError(f"Unknown dist_kernel={cfg.dist_kernel!r}")
        else:
            self.emb_pair_dist = None

        if cfg.use_pair_outer:
            self.pair_outer_a = nn.Linear(dim, pair_dim, bias=False)
            self.pair_outer_b = nn.Linear(dim, pair_dim, bias=False)
        else:
            self.pair_outer_a = None
            self.pair_outer_b = None
        self.pair_norm = nn.LayerNorm(pair_dim)

        self.t_embedder = TimestepEmbedder(hidden_dim=dim)
        self.cell_embedder = CellEmbedder(hidden_dim=dim) if cfg.use_cell_embed else None

        if cfg.use_cross_attn:
            self.backbone = DiTCrossAttn(
                dim=dim,
                depth=cfg.depth,
                num_heads=cfg.num_heads,
                pair_dim=pair_dim,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                activation_checkpointing=cfg.activation_checkpointing,
            )
        else:
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
        prev_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # tokens: (B, N, dim)
        x_t = pos + delta_t
        tokens = (
            self.atom_embed(atomic_numbers.clamp_max(self.cfg.num_elements - 1))
            + self.pos_proj(pos)
            + self.xt_proj(x_t)
        )
        if self.tag_embed is not None:
            tokens = tokens + self.tag_embed(tags.clamp(0, self.cfg.num_tags - 1))
        if self.movable_embed is not None:
            tokens = tokens + self.movable_embed(movable_mask.long())
        if self.cell_embedder is not None:
            tokens = tokens + self.cell_embedder(cell).unsqueeze(1)  # (B, 1, dim)
        if self.prev_pred_proj is not None:
            prev = prev_pred if prev_pred is not None else torch.zeros_like(pos)
            tokens = tokens + self.prev_pred_proj(prev)
        return tokens * pad_mask.unsqueeze(-1).to(tokens.dtype)

    def _pair_diff(self, pos: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        if self.cfg.pair_pbc == "3d":
            return _pair_diff_mic(pos, cell)
        if self.cfg.pair_pbc == "2d":
            # Wrap only in-plane axes (a, b); leave c alone so slab vacuum
            # along z does not produce spurious wraps.
            B, N, _ = pos.shape
            diff = pos.unsqueeze(1) - pos.unsqueeze(2)  # (B, N, N, 3)
            flat = diff.reshape(B, N * N, 3)
            cell_inv = torch.linalg.inv(cell)
            frac = torch.einsum("bnj,bjk->bnk", flat, cell_inv)  # (B, N*N, 3)
            wrap = torch.zeros_like(frac)
            wrap[..., :2] = torch.round(frac[..., :2])
            frac = frac - wrap
            mic = torch.einsum("bnj,bjk->bnk", frac, cell)
            return mic.reshape(B, N, N, 3)
        raise ValueError(f"Unknown pair_pbc={self.cfg.pair_pbc!r}")

    def _build_pair_features(
        self,
        pos: torch.Tensor,
        tags: torch.Tensor,
        pad_mask: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        # pair: (B, N, N, pair_dim)
        diff = self._pair_diff(pos, cell)
        dist2 = (diff * diff).sum(dim=-1, keepdim=True)

        pad_pair = pad_mask.unsqueeze(2) & pad_mask.unsqueeze(1)
        if self.cfg.pair_scope == "non_bulk":
            non_bulk = (tags >= 1)
            scope = pad_pair & non_bulk.unsqueeze(2) & non_bulk.unsqueeze(1)
        elif self.cfg.pair_scope == "all":
            scope = pad_pair
        else:
            raise ValueError(f"Unknown pair_scope={self.cfg.pair_scope!r}")
        v = scope.to(pos.dtype).unsqueeze(-1)

        pair = self.emb_pair_mask(v) * v
        if self.emb_pair_pos is not None:
            pair = pair + self.emb_pair_pos(diff) * v
        if self.emb_pair_dist is not None:
            if self.cfg.dist_kernel == "reciprocal":
                dist_feat = 1.0 / (1.0 + dist2)
            else:  # gaussian RBF
                d = dist2.clamp_min(1e-12).sqrt()  # (..., 1)
                centers = self._dist_rbf_centers.view(*([1] * (d.dim() - 1)), -1)
                width = (self.cfg.dist_rbf_cutoff / max(self.cfg.dist_rbf_num - 1, 1)) ** 2
                dist_feat = torch.exp(-((d - centers) ** 2) / max(width, 1e-6))
            pair = pair + self.emb_pair_dist(dist_feat) * v
        if self.emb_pair_ads is not None:
            ads = (tags == 2)
            ads_pair = (ads.unsqueeze(2) & ads.unsqueeze(1)).to(pos.dtype).unsqueeze(-1)
            pair = pair + self.emb_pair_ads(ads_pair) * v
        if self.emb_pair_ads_surf is not None:
            ads = (tags == 2)
            surf = (tags == 1)
            ads_surf = (
                (ads.unsqueeze(2) & surf.unsqueeze(1))
                | (surf.unsqueeze(2) & ads.unsqueeze(1))
            ).to(pos.dtype).unsqueeze(-1)
            pair = pair + self.emb_pair_ads_surf(ads_surf) * v
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
        prev_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not torch.isfinite(pos).all():
            raise RuntimeError("Non-finite values in pos")
        if not torch.isfinite(delta_t).all():
            raise RuntimeError("Non-finite values in delta_t")

        tokens = self._embed_tokens(
            pos=pos, delta_t=delta_t, atomic_numbers=atomic_numbers,
            tags=tags, movable_mask=movable_mask, pad_mask=pad_mask, cell=cell,
            prev_pred=prev_pred,
        )  # (B, N, dim)

        pair_pos = (pos + delta_t) if self.cfg.use_dynamic_pair_dist else pos
        pair = self._build_pair_features(pair_pos, tags, pad_mask, cell)  # (B, N, N, pair_dim)

        if self.pair_outer_a is not None:
            a = self.pair_outer_a(tokens)  # (B, N, pair_dim)
            b = self.pair_outer_b(tokens)  # (B, N, pair_dim)
            pair = pair + a.unsqueeze(2) + b.unsqueeze(1)
        pair = self.pair_norm(pair)

        c = self.t_embedder(t)  # (B, dim)

        if self.cfg.use_cross_attn:
            # Two-stream: queries=ads (updates only there), self-attn keys=ads,
            # cross-attn keys=surf (tag==1). SDPA returns NaN when every key
            # in a row is masked, so if a sample has no surface atoms we fall
            # that sample back to any non-ads real atom (bulk + padding-safe).
            ads = movable_mask & pad_mask
            surf = (tags == 1) & pad_mask
            no_surf = ~surf.any(dim=1, keepdim=True)  # (B, 1)
            surf = surf | (no_surf & pad_mask & ~ads)
            x = self.backbone(
                tokens, c,
                sa_mask=ads,
                ca_kv_mask=surf,
                update_mask=ads,
                pair_feats=pair,
            )
        else:
            x = self.backbone(tokens, c, pad_mask, pair)  # (B, N, dim)

        x = self.out_norm(x)
        out = self.out_proj(x)  # (B, N, 3)

        movable_f = movable_mask.unsqueeze(-1).to(out.dtype)
        out = out * movable_f

        if self.training and not torch.isfinite(out).all():
            raise RuntimeError("NaN detected in DiTDenoiserV2 forward")
        return out
