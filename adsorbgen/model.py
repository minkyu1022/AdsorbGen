"""DiTDenoiser: encoder-trunk-decoder flow matching model on dense padded batches.

Architecture follows AtomMOF's three-stage DiT design:
    1. AtomAttentionEncoder: shallow DiT with pair bias (atom_s, atom_z)
    2. TokenTransformer trunk: deep DiT with pair bias (token_s, token_z)
    3. AtomAttentionDecoder: shallow DiT without pair bias (atom_s)

Atom-to-token projection is a simple Linear (no block aggregation),
matching AtomMOF's current implementation where atom count == token count.

Forward signature (training):
    pred_delta1 = model(
        pos,           # (B, N, 3) Cartesian x_ref in Angstroms
        delta_t,       # (B, N, 3) current noisy displacement
        t,             # (B,) timestep in [eps, 1-eps]
        atomic_numbers,# (B, N) long
        tags,          # (B, N) long in {0, 1, 2}
        movable_mask,  # (B, N) bool
        pad_mask,      # (B, N) bool, True = real atom
        cell,          # (B, 3, 3)
        delta_e,       # (B,) float, adsorption energy condition (pretraining: 0)
        cond_drop,     # (B,) bool, True = drop condition (CFG / pretrain)
    ) -> (B, N, 3) pred_delta1  (zeroed on non-movable and padding atoms)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from adsorbgen.flow import minimum_image
from adsorbgen.transformer import DiT, TimestepEmbedder


@dataclass
class DiTDenoiserConfig:
    # Encoder / decoder dimensions
    atom_s: int = 256
    atom_z: int = 128
    # Trunk dimensions
    token_s: int = 512
    token_z: int = 256
    # Depths
    enc_depth: int = 2
    trunk_depth: int = 12
    dec_depth: int = 2
    # Heads
    enc_heads: int = 4
    trunk_heads: int = 8
    dec_heads: int = 4
    # Shared
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    num_elements: int = 100
    num_tags: int = 3
    sigma: float = 0.5
    delta_e_max: float = 2.0
    delta_e_freq_dim: int = 256
    activation_checkpointing: bool = False


class DeltaEEmbedder(nn.Module):
    """ΔE -> d-dim vector via sinusoidal frequency + MLP. fc2 is zero-init."""

    def __init__(self, hidden_dim: int, frequency_embedding_dim: int = 256, delta_e_max: float = 2.0):
        super().__init__()
        self.frequency_embedding_dim = frequency_embedding_dim
        self.delta_e_max = delta_e_max
        self.fc1 = nn.Linear(frequency_embedding_dim, hidden_dim, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @staticmethod
    def _sinusoid(e: torch.Tensor, dim: int, max_period: float = 10.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=e.device) / half
        )
        args = e[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, delta_e: torch.Tensor) -> torch.Tensor:
        clamped = delta_e.clamp(min=0.0, max=self.delta_e_max)
        freq = self._sinusoid(clamped, self.frequency_embedding_dim)
        return self.fc2(self.act(self.fc1(freq)))


class CellEmbedder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    def forward(self, cell: torch.Tensor) -> torch.Tensor:
        return self.net(cell.reshape(cell.shape[0], 9))


def _pair_diff_mic(pos: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Pairwise MIC displacements ``pos_j - pos_i`` with shape (B, N, N, 3)."""
    B, N, _ = pos.shape
    diff = pos.unsqueeze(1) - pos.unsqueeze(2)  # (B, N, N, 3); diff[b,i,j] = pos_j - pos_i
    diff_flat = diff.reshape(B, N * N, 3)
    mic = minimum_image(diff_flat, cell)
    return mic.reshape(B, N, N, 3)


class DiTDenoiser(nn.Module):
    def __init__(self, cfg: DiTDenoiserConfig):
        super().__init__()
        self.cfg = cfg
        atom_s = cfg.atom_s
        atom_z = cfg.atom_z
        token_s = cfg.token_s
        token_z = cfg.token_z

        # ── Input embeddings (atom_s) ──
        self.atom_embed = nn.Embedding(cfg.num_elements, atom_s)
        self.tag_embed = nn.Embedding(cfg.num_tags, atom_s)
        self.movable_embed = nn.Embedding(2, atom_s)
        self.pos_proj = nn.Linear(3, atom_s, bias=True)
        self.xt_proj = nn.Linear(3, atom_s, bias=True)

        # ── Pair features (atom_z) ──
        self.emb_pair_pos = nn.Linear(3, atom_z, bias=False)
        self.emb_pair_dist = nn.Linear(1, atom_z, bias=False)
        self.emb_pair_mask = nn.Linear(1, atom_z, bias=False)
        self.emb_pair_ads = nn.Linear(1, atom_z, bias=False)

        # ── Condition embeddings (atom_s) ──
        self.t_embedder = TimestepEmbedder(hidden_dim=atom_s)
        self.delta_e_embedder = DeltaEEmbedder(
            hidden_dim=atom_s,
            frequency_embedding_dim=cfg.delta_e_freq_dim,
            delta_e_max=cfg.delta_e_max,
        )
        self.cell_embedder = CellEmbedder(hidden_dim=atom_s)

        # ── Encoder (atom_s, atom_z) ──
        self.encoder = DiT(
            dim=atom_s, depth=cfg.enc_depth, num_heads=cfg.enc_heads,
            pair_dim=atom_z, mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
            activation_checkpointing=cfg.activation_checkpointing,
        )

        # ── Atom -> Token projections ──
        self.atom_to_token = nn.Sequential(
            nn.Linear(atom_s, token_s, bias=False), nn.ReLU(),
        )
        self.atom_to_token_pair = nn.Sequential(
            nn.Linear(atom_z, token_z, bias=False), nn.ReLU(),
        )
        self.cond_proj = nn.Linear(atom_s, token_s, bias=False)

        # ── Trunk initialization (AtomMOF-style) ──
        self.trunk_s_init = nn.Linear(token_s, token_s, bias=False)
        self.trunk_z_init_1 = nn.Linear(token_s, token_z, bias=False)
        self.trunk_z_init_2 = nn.Linear(token_s, token_z, bias=False)
        self.trunk_s_mlp = nn.Sequential(
            nn.LayerNorm(token_s), nn.Linear(token_s, token_s, bias=False),
        )
        self.trunk_z_mlp = nn.Sequential(
            nn.LayerNorm(token_z), nn.Linear(token_z, token_z, bias=False),
        )

        # ── Trunk (token_s, token_z) ──
        self.trunk = DiT(
            dim=token_s, depth=cfg.trunk_depth, num_heads=cfg.trunk_heads,
            pair_dim=token_z, mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
            activation_checkpointing=cfg.activation_checkpointing,
        )

        # ── Token -> Atom projection ──
        self.token_to_atom = nn.Sequential(
            nn.Linear(token_s, atom_s, bias=False), nn.ReLU(),
        )

        # ── Decoder (atom_s, no pair) ──
        self.decoder = DiT(
            dim=atom_s, depth=cfg.dec_depth, num_heads=cfg.dec_heads,
            pair_dim=0, mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
            activation_checkpointing=cfg.activation_checkpointing,
        )

        # ── Output head (zero-init) ──
        self.out_norm = nn.LayerNorm(atom_s, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(atom_s, 3, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ── helpers ──

    def _encode_tokens(
        self,
        pos: torch.Tensor,
        delta_t: torch.Tensor,
        atomic_numbers: torch.Tensor,
        tags: torch.Tensor,
        movable_mask: torch.Tensor,
        pad_mask: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        x_t = pos + delta_t
        cell_emb = self.cell_embedder(cell).unsqueeze(1)  # (B, 1, atom_s)
        tokens = (
            self.atom_embed(atomic_numbers.clamp_max(self.cfg.num_elements - 1))
            + self.tag_embed(tags.clamp(0, self.cfg.num_tags - 1))
            + self.movable_embed(movable_mask.long())
            + self.pos_proj(pos)
            + self.xt_proj(x_t)
            + cell_emb
        )
        tokens = tokens * pad_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens  # (B, N, atom_s)

    def _build_pair_features(
        self,
        pos: torch.Tensor,
        tags: torch.Tensor,
        pad_mask: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """Build (B, N, N, atom_z) pair features on the non-bulk block."""
        diff = _pair_diff_mic(pos, cell)  # (B, N, N, 3)
        dist2 = (diff * diff).sum(dim=-1, keepdim=True)  # (B, N, N, 1)
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
        return pair  # (B, N, N, atom_z)

    # ── forward ──

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
        delta_e: Optional[torch.Tensor] = None,
        cond_drop: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.isfinite(pos).all():
            raise RuntimeError("Non-finite values in pos")
        if not torch.isfinite(delta_t).all():
            raise RuntimeError("Non-finite values in delta_t")

        B = pos.shape[0]

        # 1. Input embeddings
        tokens = self._encode_tokens(
            pos=pos, delta_t=delta_t, atomic_numbers=atomic_numbers,
            tags=tags, movable_mask=movable_mask, pad_mask=pad_mask,
            cell=cell,
        )  # (B, N, atom_s)
        pair_feats = self._build_pair_features(pos, tags, pad_mask, cell)  # (B, N, N, atom_z)

        # 2. Build condition vector at atom_s dim (time + ΔE only; cell is per-token)
        c = self.t_embedder(t)
        if delta_e is None:
            delta_e = torch.zeros(B, device=pos.device, dtype=pos.dtype)
        e_cond = self.delta_e_embedder(delta_e)
        if cond_drop is not None:
            drop_mask = cond_drop.view(B, 1).to(e_cond.dtype)
            e_cond = e_cond * (1.0 - drop_mask)
        c = c + e_cond  # (B, atom_s)

        # 3. Encoder
        x = self.encoder(tokens, c, pad_mask, pair_feats)  # (B, N, atom_s)

        # 4. Atom -> Token
        token_single = self.atom_to_token(x)  # (B, N, token_s)
        token_pair = self.atom_to_token_pair(pair_feats)  # (B, N, N, token_z)
        c_trunk = self.cond_proj(c)  # (B, token_s)

        # 5. Trunk initialization (outer-product pair enrichment)
        s_init = self.trunk_s_init(token_single)  # (B, N, token_s)
        z_init = token_pair + self.trunk_z_init_1(s_init).unsqueeze(2) + self.trunk_z_init_2(s_init).unsqueeze(1)
        s = self.trunk_s_mlp(s_init)  # (B, N, token_s)
        z = self.trunk_z_mlp(z_init)  # (B, N, N, token_z)

        # 6. Trunk
        x_trunk = self.trunk(s, c_trunk, pad_mask, z)  # (B, N, token_s)

        # 7. Token -> Atom (residual)
        x = x + self.token_to_atom(x_trunk)  # (B, N, atom_s)

        # 8. Decoder (no pair features)
        x = self.decoder(x, c, pad_mask)  # (B, N, atom_s)

        # 9. Output
        x = self.out_norm(x)
        out = self.out_proj(x)  # (B, N, 3)

        movable_f = movable_mask.unsqueeze(-1).to(out.dtype)
        out = out * movable_f

        if self.training and not torch.isfinite(out).all():
            raise RuntimeError("NaN detected in DiTDenoiser forward")
        return out
