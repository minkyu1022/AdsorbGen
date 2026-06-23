"""DiTDenoiser: encoder-trunk-decoder flow matching model on dense padded batches.

Architecture follows AtomMOF's three-stage DiT design:
    1. AtomAttentionEncoder: shallow DiT with pair bias (atom_s, atom_z)
    2. TokenTransformer trunk: deep DiT with pair bias (token_s, token_z)
    3. AtomAttentionDecoder: shallow DiT without pair bias (atom_s)

Atom-to-token projection is a simple Linear (no block aggregation), matching
AtomMOF where atom count == token count.

Parametrisation: AtomMOF-style absolute-coordinate flow matching with direct
x_1 prediction.
    x_0 = structured prior (LMDB pos_init + fairchem placement)
    x_1 = pos_relaxed
    x_t = (1-t) x_0 + t x_1   (movable atoms only; non-movable stay at x_0)
    Model output: pred_x_1   (absolute coords of final state)
    Output head is zero-inited; initial pred = 0 for movable atoms. Non-movable
    atoms and padding are held at x_0 / zero.

Forward signature (training):
    pred_x_1 = model(
        pos,            # (B, N, 3) x_0 in Angstroms (static anchor)
        x_t,            # (B, N, 3) current interpolated state
        t,              # (B,) timestep in [eps, 1-eps]
        atomic_numbers, # (B, N) long
        tags,           # (B, N) long in {0, 1, 2}
        movable_mask,   # (B, N) bool
        pad_mask,       # (B, N) bool, True = real atom
        cell,           # (B, 3, 3)
    ) -> (B, N, 3) pred_x_1 (non-movable atoms held at pos; padding zeroed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from adsorbgen.flow import minimum_image
from adsorbgen.models.transformer import DiT, TimestepEmbedder


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
    activation_checkpointing: bool = False
    # When True, skip the outer-product pair enrichment at encoder→trunk
    # transition (step 5). Pair is just linearly projected without enrichment.
    skip_trunk_enrich: bool = False
    # Ablation: give the decoder AttentionPairBias like the encoder/trunk.
    # Uses atom-level pair_feats (atom_z). Default False = original v1.
    dec_pair_bias: bool = False
    # Ablation: replace atom_to_token / atom_to_token_pair / token_to_atom
    # Linear+ReLU bridges with Identity. Requires atom_s==token_s and
    # atom_z==token_z. Isolates the "inter-stage gating" axis of v1→v2.
    skip_stage_gates: bool = False
    # Distance kernel for pair feature: "reciprocal" = 1/(1+d^2) (original v1);
    # "gaussian" = sum_k exp(-(d-mu_k)^2/width^2) RBF (ported from v2).
    dist_kernel: str = "reciprocal"
    dist_rbf_num: int = 16
    dist_rbf_cutoff: float = 6.0
    # CatFlow-style adsorbate reference geometry: when True, the model takes
    # an extra per-atom (B, N, 3) field ``ads_ref_pos`` that holds the
    # centered canonical molecular geometry on tag==2 atoms (zeros elsewhere)
    # and projects it into the per-atom embedding. Tells the model the
    # canonical bond pattern of the adsorbate, so it has a structural prior
    # against dissociating it.
    use_ads_ref_pos: bool = False
    # When True, pair features are computed from the noisy x_t instead of the
    # static x_0 (pos). The original v1 always used x_0, so x_t was only seen
    # by single-atom features (xt_proj). Ported from v2's v9-dynamic-pair.
    use_dynamic_pair_dist: bool = False
    # When True, pair displacements use the minimum-image convention (default,
    # matches the original v1). Set False to use raw cartesian differences —
    # useful only for ablation when paired with use_dynamic_pair_dist=True so
    # the model sees absolute relative geometry without periodic wrapping.
    pair_use_mic: bool = True
    # CatFlow-inspired output ablation: use a separate zero-initialized
    # coordinate projection for tag==2 adsorbate atoms while keeping the
    # decoder shared. This isolates whether the final coordinate basis for
    # molecule atoms conflicts with the surface-relaxation head.
    use_ads_specific_head: bool = False
    # AdsorbSample-inspired pair conditioning while keeping the AdsorbGen DiT
    # backbone: add pair-type, adsorbate bond, and adsorbate graph-topology
    # embeddings through branch-wise projections plus learned gated fusion.
    # This preserves flexible surface prediction and the current supervised
    # x1 objective.
    use_typed_pair_features: bool = False
    # When True, pair bias also covers movable/bulk context pairs instead of
    # only surface/adsorbate pairs. This lets flexible surface atoms attend to
    # fixed subsurface geometry without making bulk atoms movable.
    typed_pair_include_bulk: bool = False
    max_topological_distance: int = 8
    ads_bond_factor: float = 1.25
    # Coordinate preconditioning for the v1/AtomMOF-style absolute-x1 model.
    # Pair distances, MIC, RBF cutoffs, and losses stay in raw Angstrom space;
    # only the linear coordinate projections and output head are wrapped.
    coord_norm_mode: str = "none"
    coord_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    coord_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ads_ref_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ads_ref_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # CatFlow-style adsorbate output factorization. When enabled, surface
    # atoms still use the regular coordinate head, while adsorbate atoms are
    # reconstructed from a pooled center head plus a per-atom rel-pos head.
    use_ads_center_rel_head: bool = False
    ads_center_mean: tuple[float, float, float] = (-0.4249, -0.0863, 5.8592)
    ads_center_scale: tuple[float, float, float] = (2.9450, 3.2665, 1.7597)
    ads_rel_pos_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ads_rel_pos_scale: tuple[float, float, float] = (0.6319, 0.8760, 0.7194)
    ads_rel_output_sum_zero: bool = False
    # Pair-level 3D RoPE over attention logits. Coordinate representation,
    # flow target, and losses remain absolute Cartesian.
    use_pair_rope: bool = False
    pair_rope_dim: int = 48
    pair_rope_base: float = 10.0
    # Langevin parametrization: add learned scaling times detached MLIP force.
    # scale_mode="global" is one scalar per structure, "atom" is one scalar
    # per atom, and "vector" is one scalar per atom/xyz component.
    use_langevin_param: bool = False
    langevin_force_clip: float = 100.0
    langevin_eval_on: str = "x_t"
    langevin_scale_mode: str = "global"
    langevin_use_ads_specific_head: bool = False
    # OMatG-style stochastic-interpolant denoiser branch.  When enabled, the
    # model predicts both the SI velocity/x1 head and eta(t, x_t), the latent
    # Gaussian noise used in x_t = I(t, x0, x1) + gamma(t) z.
    use_si_denoiser: bool = False
    si_denoiser_use_ads_specific_head: bool = False


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


def _pair_diff_raw(pos: torch.Tensor) -> torch.Tensor:
    """Pairwise raw displacements ``pos_j - pos_i`` (no MIC)."""
    return pos.unsqueeze(1) - pos.unsqueeze(2)  # (B, N, N, 3)


_COVALENT_RADII = {
    1: 0.31, 5: 0.85, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39,
}


def _lookup_covalent_radii(numbers: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    radii = torch.full_like(numbers, 0.8, dtype=dtype)
    for z, r in _COVALENT_RADII.items():
        radii = torch.where(numbers == int(z), torch.full_like(radii, float(r)), radii)
    return radii


def _infer_ads_bond_from_ref(
    atomic_numbers: torch.Tensor,
    tags: torch.Tensor,
    pad_mask: torch.Tensor,
    ads_ref_pos: torch.Tensor,
    bond_factor: float,
) -> torch.Tensor:
    """Infer adsorbate covalent bonds from centered reference geometry."""
    radii = _lookup_covalent_radii(atomic_numbers, ads_ref_pos.dtype)
    diff = ads_ref_pos.unsqueeze(1) - ads_ref_pos.unsqueeze(2)
    dist = torch.linalg.norm(diff, dim=-1)
    limit = float(bond_factor) * (radii.unsqueeze(1) + radii.unsqueeze(2))
    ads = tags == 2
    pair = ads.unsqueeze(1) & ads.unsqueeze(2) & pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)
    n = tags.shape[1]
    not_self = ~torch.eye(n, device=tags.device, dtype=torch.bool).unsqueeze(0)
    return ((dist > 0.1) & (dist <= limit) & pair & not_self).to(ads_ref_pos.dtype)


def _topological_distances(
    bond: torch.Tensor,
    tags: torch.Tensor,
    pad_mask: torch.Tensor,
    max_dist: int,
) -> torch.Tensor:
    """Shortest-path distances on the adsorbate graph, scattered into NxN."""
    B, N, _ = bond.shape
    out = torch.zeros((B, N, N), device=bond.device, dtype=torch.long)
    with torch.no_grad():
        for b in range(B):
            idx = torch.nonzero((tags[b] == 2) & pad_mask[b], as_tuple=False).flatten()
            m = int(idx.numel())
            if m <= 1:
                continue
            adj = bond[b].index_select(0, idx).index_select(1, idx) > 0
            inf = torch.full((m, m), int(max_dist) + 1, device=bond.device, dtype=torch.long)
            dist = torch.where(adj, torch.ones_like(inf), inf)
            eye = torch.eye(m, device=bond.device, dtype=torch.bool)
            dist = torch.where(eye, torch.zeros_like(dist), dist)
            for k in range(m):
                dist = torch.minimum(dist, dist[:, k:k + 1] + dist[k:k + 1, :])
            dist = torch.where(dist > int(max_dist), torch.zeros_like(dist), dist)
            out[b].index_put_((idx[:, None], idx[None, :]), dist)
    return out


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
        if cfg.use_ads_center_rel_head:
            self.ads_center_xt_proj = nn.Linear(3, atom_s, bias=True)
            self.ads_rel_xt_proj = nn.Linear(3, atom_s, bias=True)
        # CatFlow-style ads-reference geometry projection (only on tag==2 atoms).
        if cfg.use_ads_ref_pos:
            self.ads_ref_proj = nn.Linear(3, atom_s, bias=False)

        # ── Pair features (atom_z) ──
        self.emb_pair_pos = nn.Linear(3, atom_z, bias=False)
        if cfg.dist_kernel == "reciprocal":
            self.emb_pair_dist = nn.Linear(1, atom_z, bias=False)
            self.register_buffer("_dist_rbf_centers", torch.empty(0), persistent=False)
        elif cfg.dist_kernel == "gaussian":
            centers = torch.linspace(0.0, cfg.dist_rbf_cutoff, cfg.dist_rbf_num)
            self.register_buffer("_dist_rbf_centers", centers, persistent=False)
            self.emb_pair_dist = nn.Linear(cfg.dist_rbf_num, atom_z, bias=False)
        else:
            raise ValueError(f"Unknown dist_kernel={cfg.dist_kernel!r}")
        self.emb_pair_mask = nn.Linear(1, atom_z, bias=False)
        self.emb_pair_ads = nn.Linear(1, atom_z, bias=False)
        if cfg.use_typed_pair_features:
            dist_dim = 1 if cfg.dist_kernel == "reciprocal" else cfg.dist_rbf_num
            self.typed_pair_branch_names = (
                "ads_ads",
                "ads_bond",
                "ads_surface",
                "surface_surface",
                "surface_bulk",
            )
            branch_in = 3 + dist_dim + 1
            self.typed_branch_proj = nn.ModuleDict({
                name: nn.Linear(branch_in, atom_z, bias=False)
                for name in self.typed_pair_branch_names
            })
            self.typed_pair_gate = nn.Sequential(
                nn.Linear(dist_dim + len(self.typed_pair_branch_names), atom_z),
                nn.SiLU(),
                nn.Linear(atom_z, len(self.typed_pair_branch_names)),
            )
            self.typed_topology_embed = nn.Embedding(
                cfg.max_topological_distance + 1, atom_z, padding_idx=0,
            )
            self.typed_bond_embed = nn.Embedding(2, atom_z, padding_idx=0)
            # Start exactly from the old DiT pair-bias behavior. The branch
            # features learn in as a residual without perturbing step 0.
            for proj in self.typed_branch_proj.values():
                nn.init.zeros_(proj.weight)
            nn.init.zeros_(self.typed_topology_embed.weight)
            nn.init.zeros_(self.typed_bond_embed.weight)

        # ── Atom-level single→pair enrichment (AtomMOF c_to_p_trans_q/k) ──
        # Two independent linear projections inject single features into the
        # pair grid along i-axis and j-axis. Single here is the full token
        # embedding (including x_t), so pair indirectly perceives x_t.
        self.pair_from_single_i = nn.Linear(atom_s, atom_z, bias=False)
        self.pair_from_single_j = nn.Linear(atom_s, atom_z, bias=False)

        # ── Condition embeddings (atom_s) ──
        self.t_embedder = TimestepEmbedder(hidden_dim=atom_s)
        self.cell_embedder = CellEmbedder(hidden_dim=atom_s)
        if cfg.use_langevin_param:
            if cfg.langevin_scale_mode not in {"global", "atom", "vector"}:
                raise ValueError(
                    "langevin_scale_mode must be one of "
                    "{'global', 'atom', 'vector'}"
                )
            out_dim = 1 if cfg.langevin_scale_mode in {"global", "atom"} else 3
            self.langevin_scale = nn.Linear(atom_s, out_dim, bias=True)
            nn.init.zeros_(self.langevin_scale.weight)
            nn.init.zeros_(self.langevin_scale.bias)
            if cfg.langevin_use_ads_specific_head:
                if cfg.langevin_scale_mode == "global":
                    raise ValueError(
                        "langevin_use_ads_specific_head requires "
                        "langevin_scale_mode='atom' or 'vector'"
                    )
                self.ads_langevin_scale = nn.Linear(atom_s, out_dim, bias=True)
                nn.init.zeros_(self.ads_langevin_scale.weight)
                nn.init.zeros_(self.ads_langevin_scale.bias)

        # ── Encoder (atom_s, atom_z) ──
        self.encoder = DiT(
            dim=atom_s, depth=cfg.enc_depth, num_heads=cfg.enc_heads,
            pair_dim=atom_z, mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
            activation_checkpointing=cfg.activation_checkpointing,
            pair_rope_dim=cfg.pair_rope_dim if cfg.use_pair_rope else 0,
            pair_rope_base=cfg.pair_rope_base,
        )

        # ── Atom -> Token projections ──
        if cfg.skip_stage_gates:
            assert atom_s == token_s and atom_z == token_z, (
                "skip_stage_gates requires atom_s==token_s and atom_z==token_z"
            )
            self.atom_to_token = nn.Identity()
            self.atom_to_token_pair = nn.Identity()
        else:
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
            pair_rope_dim=cfg.pair_rope_dim if cfg.use_pair_rope else 0,
            pair_rope_base=cfg.pair_rope_base,
        )

        # ── Token -> Atom projection ──
        if cfg.skip_stage_gates:
            self.token_to_atom = nn.Identity()
        else:
            self.token_to_atom = nn.Sequential(
                nn.Linear(token_s, atom_s, bias=False), nn.ReLU(),
            )

        # ── Decoder (atom_s, optional atom-level pair bias) ──
        dec_pair_dim = atom_z if cfg.dec_pair_bias else 0
        self.decoder = DiT(
            dim=atom_s, depth=cfg.dec_depth, num_heads=cfg.dec_heads,
            pair_dim=dec_pair_dim, mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
            activation_checkpointing=cfg.activation_checkpointing,
            pair_rope_dim=cfg.pair_rope_dim if cfg.use_pair_rope else 0,
            pair_rope_base=cfg.pair_rope_base,
        )

        # ── Output head (zero-init) ──
        self.out_norm = nn.LayerNorm(atom_s, elementwise_affine=False, eps=1e-6)
        self.coord_out = nn.Linear(atom_s, 3, bias=True)
        nn.init.zeros_(self.coord_out.weight)
        nn.init.zeros_(self.coord_out.bias)
        if cfg.use_ads_specific_head:
            self.ads_coord_out = nn.Linear(atom_s, 3, bias=True)
            nn.init.zeros_(self.ads_coord_out.weight)
            nn.init.zeros_(self.ads_coord_out.bias)
        if cfg.use_ads_center_rel_head:
            self.ads_center_out = nn.Linear(atom_s, 3, bias=True)
            self.ads_rel_pos_out = nn.Linear(atom_s, 3, bias=True)
            nn.init.zeros_(self.ads_center_out.weight)
            nn.init.zeros_(self.ads_center_out.bias)
            nn.init.zeros_(self.ads_rel_pos_out.weight)
            nn.init.zeros_(self.ads_rel_pos_out.bias)
        if cfg.use_si_denoiser:
            self.eta_out = nn.Linear(atom_s, 3, bias=True)
            nn.init.zeros_(self.eta_out.weight)
            nn.init.zeros_(self.eta_out.bias)
            if cfg.si_denoiser_use_ads_specific_head:
                self.ads_eta_out = nn.Linear(atom_s, 3, bias=True)
                nn.init.zeros_(self.ads_eta_out.weight)
                nn.init.zeros_(self.ads_eta_out.bias)

        self.register_buffer(
            "_coord_mean",
            torch.tensor(cfg.coord_mean, dtype=torch.float32).view(1, 1, 3),
            persistent=False,
        )
        self.register_buffer(
            "_coord_scale",
            torch.tensor(cfg.coord_scale, dtype=torch.float32).clamp_min(1e-6).view(1, 1, 3),
            persistent=False,
        )
        self.register_buffer(
            "_ads_ref_mean",
            torch.tensor(cfg.ads_ref_mean, dtype=torch.float32).view(1, 1, 3),
            persistent=False,
        )
        self.register_buffer(
            "_ads_ref_scale",
            torch.tensor(cfg.ads_ref_scale, dtype=torch.float32).clamp_min(1e-6).view(1, 1, 3),
            persistent=False,
        )
        self.register_buffer(
            "_ads_center_mean",
            torch.tensor(cfg.ads_center_mean, dtype=torch.float32).view(1, 3),
            persistent=False,
        )
        self.register_buffer(
            "_ads_center_scale",
            torch.tensor(cfg.ads_center_scale, dtype=torch.float32).clamp_min(1e-6).view(1, 3),
            persistent=False,
        )
        self.register_buffer(
            "_ads_rel_pos_mean",
            torch.tensor(cfg.ads_rel_pos_mean, dtype=torch.float32).view(1, 1, 3),
            persistent=False,
        )
        self.register_buffer(
            "_ads_rel_pos_scale",
            torch.tensor(cfg.ads_rel_pos_scale, dtype=torch.float32).clamp_min(1e-6).view(1, 1, 3),
            persistent=False,
        )

    # ── helpers ──

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Backward compatibility for checkpoints saved before output heads were
        # renamed from *_proj to *_out.
        aliases = {
            "out_proj": "coord_out",
            "ads_out_proj": "ads_coord_out",
            "ads_center_out_proj": "ads_center_out",
            "ads_rel_out_proj": "ads_rel_pos_out",
        }
        for old_name, new_name in aliases.items():
            for suffix in ("weight", "bias"):
                old_key = f"{prefix}{old_name}.{suffix}"
                new_key = f"{prefix}{new_name}.{suffix}"
                if old_key in state_dict:
                    if new_key not in state_dict:
                        state_dict[new_key] = state_dict[old_key]
                    state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _normalize_coord(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._coord_mean.to(dtype=x.dtype, device=x.device)) / self._coord_scale.to(dtype=x.dtype, device=x.device)

    def _denormalize_coord(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._coord_scale.to(dtype=x.dtype, device=x.device) + self._coord_mean.to(dtype=x.dtype, device=x.device)

    def _normalize_ads_ref(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._ads_ref_mean.to(dtype=x.dtype, device=x.device)) / self._ads_ref_scale.to(dtype=x.dtype, device=x.device)

    def _normalize_ads_center(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._ads_center_mean.to(dtype=x.dtype, device=x.device)) / self._ads_center_scale.to(dtype=x.dtype, device=x.device)

    def _denormalize_ads_center(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._ads_center_scale.to(dtype=x.dtype, device=x.device) + self._ads_center_mean.to(dtype=x.dtype, device=x.device)

    def _normalize_ads_rel_pos(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._ads_rel_pos_mean.to(dtype=x.dtype, device=x.device)) / self._ads_rel_pos_scale.to(dtype=x.dtype, device=x.device)

    def _denormalize_ads_rel_pos(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._ads_rel_pos_scale.to(dtype=x.dtype, device=x.device) + self._ads_rel_pos_mean.to(dtype=x.dtype, device=x.device)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        w = mask.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)

    def _encode_tokens(
        self,
        pos: torch.Tensor,
        x_t: torch.Tensor,
        atomic_numbers: torch.Tensor,
        tags: torch.Tensor,
        movable_mask: torch.Tensor,
        pad_mask: torch.Tensor,
        cell: torch.Tensor,
        ads_ref_pos: Optional[torch.Tensor] = None,
        ads_center_t: Optional[torch.Tensor] = None,
        ads_rel_pos_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Atom single: element + tag + movable + x_0 (pos) + x_t + cell
        (+ optional CatFlow-style ads reference geometry on tag==2 atoms).

        Both x_0 (via pos_proj) and x_t (via xt_proj) are embedded with
        independent linear maps so the model sees current state and the
        static anchor.
        """
        # Only x_t (and coordinate output heads) go through the normalization
        # wrapper. For the CatFlow-style ads split variant, adsorbate x_t is
        # exposed as normalized center + normalized rel-pos rather than a
        # single absolute coordinate projection.
        x_t_in = self._normalize_coord(x_t)
        xt_tokens = self.xt_proj(x_t_in)
        if self.cfg.use_ads_center_rel_head:
            ads_mask_bool = (tags == 2) & movable_mask & pad_mask
            ads_mask = ads_mask_bool.unsqueeze(-1).to(x_t.dtype)
            if ads_center_t is None:
                ads_center_t = self._masked_mean(x_t, ads_mask_bool, dim=1)
            if ads_rel_pos_t is None:
                ads_rel_pos_t = (x_t - ads_center_t[:, None, :]) * ads_mask
            ads_rel_pos_t = ads_rel_pos_t * ads_mask
            ads_xt_tokens = (
                self.ads_center_xt_proj(self._normalize_ads_center(ads_center_t)).unsqueeze(1)
                + self.ads_rel_xt_proj(self._normalize_ads_rel_pos(ads_rel_pos_t))
            )
            xt_tokens = ads_xt_tokens * ads_mask + xt_tokens * (1.0 - ads_mask)
        cell_emb = self.cell_embedder(cell).unsqueeze(1)  # (B, 1, atom_s)
        tokens = (
            self.atom_embed(atomic_numbers.clamp_max(self.cfg.num_elements - 1))
            + self.tag_embed(tags.clamp(0, self.cfg.num_tags - 1))
            + self.movable_embed(movable_mask.long())
            + self.pos_proj(pos)
            + xt_tokens
            + cell_emb
        )
        if self.cfg.use_ads_ref_pos:
            assert ads_ref_pos is not None, (
                "use_ads_ref_pos=True but ads_ref_pos was not passed to forward; "
                "dataset/collate must populate it."
            )
            ads_mask = (tags == 2).unsqueeze(-1).to(tokens.dtype)
            tokens = tokens + self.ads_ref_proj(ads_ref_pos) * ads_mask
        tokens = tokens * pad_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens  # (B, N, atom_s)

    def _build_pair_features(
        self,
        pos: torch.Tensor,
        atomic_numbers: torch.Tensor,
        tags: torch.Tensor,
        pad_mask: torch.Tensor,
        cell: torch.Tensor,
        ads_ref_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build (B, N, N, atom_z) pair features on the non-bulk block."""
        if self.cfg.pair_use_mic:
            diff = _pair_diff_mic(pos, cell)  # (B, N, N, 3)
        else:
            diff = _pair_diff_raw(pos)  # (B, N, N, 3)
        dist2 = (diff * diff).sum(dim=-1, keepdim=True)  # (B, N, N, 1)
        if self.cfg.dist_kernel == "reciprocal":
            dist_feat = 1.0 / (1.0 + dist2)  # (B, N, N, 1)
        else:  # gaussian RBF
            d = dist2.clamp_min(1e-12).sqrt()                    # (B, N, N, 1)
            centers = self._dist_rbf_centers.view(*([1] * (d.dim() - 1)), -1)
            width = (self.cfg.dist_rbf_cutoff / max(self.cfg.dist_rbf_num - 1, 1)) ** 2
            dist_feat = torch.exp(-((d - centers) ** 2) / max(width, 1e-6))  # (B, N, N, K)

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
        if self.cfg.use_typed_pair_features:
            pair = pair + self._build_typed_pair_features(
                diff=diff,
                dist=dist2.clamp_min(1e-12).sqrt().squeeze(-1),
                dist_feat=dist_feat,
                atomic_numbers=atomic_numbers,
                tags=tags,
                pad_mask=pad_mask,
                ads_ref_pos=ads_ref_pos,
            )
        return pair  # (B, N, N, atom_z)

    def _build_typed_pair_features(
        self,
        diff: torch.Tensor,
        dist: torch.Tensor,
        dist_feat: torch.Tensor,
        atomic_numbers: torch.Tensor,
        tags: torch.Tensor,
        pad_mask: torch.Tensor,
        ads_ref_pos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Branch-wise pair features with learned gated fusion.

        Branch semantics:
        - ads_ads: all adsorbate internal directed pairs, no cutoff
        - ads_bond: adsorbate covalent graph from ads_ref_pos, no cutoff
        - ads_surface: adsorbate/surface contacts within dist_rbf_cutoff
        - surface_surface: local surface graph within dist_rbf_cutoff
        - surface_bulk: read-only bulk context into surface within cutoff
        """
        if ads_ref_pos is None:
            raise RuntimeError(
                "use_typed_pair_features=True requires ads_ref_pos; "
                "set use_ads_ref_pos=True in the variant."
            )
        B, N, _ = tags.shape[0], tags.shape[1], diff.shape[-1]
        dtype = diff.dtype
        device = diff.device
        ads = (tags == 2) & pad_mask
        surface = (tags == 1) & pad_mask
        bulk = (tags == 0) & pad_mask
        recv_ads = ads.unsqueeze(2)
        send_ads = ads.unsqueeze(1)
        recv_surface = surface.unsqueeze(2)
        send_surface = surface.unsqueeze(1)
        recv_bulk = bulk.unsqueeze(2)
        send_bulk = bulk.unsqueeze(1)
        not_self = ~torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        radius = dist <= float(self.cfg.dist_rbf_cutoff)

        ads_ads = recv_ads & send_ads & not_self
        bond = _infer_ads_bond_from_ref(
            atomic_numbers=atomic_numbers,
            tags=tags,
            pad_mask=pad_mask,
            ads_ref_pos=ads_ref_pos,
            bond_factor=self.cfg.ads_bond_factor,
        ) > 0
        ads_surface = ((recv_ads & send_surface) | (recv_surface & send_ads)) & radius
        surface_surface = recv_surface & send_surface & radius & not_self
        surface_bulk = (
            ((recv_surface & send_bulk) | (recv_bulk & send_surface)) & radius
            if self.cfg.typed_pair_include_bulk
            else torch.zeros_like(surface_surface)
        )
        masks = [ads_ads, bond, ads_surface, surface_surface, surface_bulk]

        branch_mask = torch.stack([m.to(dtype) for m in masks], dim=-1)
        gate_in = torch.cat([dist_feat, branch_mask], dim=-1)
        gate_logits = self.typed_pair_gate(gate_in)
        gate_logits = gate_logits.masked_fill(branch_mask <= 0, -1e4)
        has_branch = branch_mask.sum(dim=-1, keepdim=True) > 0
        gates = torch.softmax(gate_logits, dim=-1) * has_branch.to(dtype)

        typed = diff.new_zeros(B, N, N, self.cfg.atom_z)
        for idx, name in enumerate(self.typed_pair_branch_names):
            m = branch_mask[..., idx:idx + 1]
            feat = torch.cat([diff, dist_feat, m], dim=-1)
            typed = typed + gates[..., idx:idx + 1] * self.typed_branch_proj[name](feat) * m

        topo = _topological_distances(
            bond.to(dtype), tags, pad_mask, self.cfg.max_topological_distance,
        ).clamp(0, self.cfg.max_topological_distance)
        typed = typed + self.typed_topology_embed(topo) * ads_ads.unsqueeze(-1).to(dtype)
        typed = typed + self.typed_bond_embed(bond.long()) * bond.unsqueeze(-1).to(dtype)
        return typed

    # ── forward ──

    def forward(
        self,
        pos: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        atomic_numbers: torch.Tensor,
        tags: torch.Tensor,
        movable_mask: torch.Tensor,
        pad_mask: torch.Tensor,
        cell: torch.Tensor,
        ads_ref_pos: Optional[torch.Tensor] = None,
        ads_center_t: Optional[torch.Tensor] = None,
        ads_rel_pos_t: Optional[torch.Tensor] = None,
        mlip_force: Optional[torch.Tensor] = None,
        langevin_prediction_type: str = "x1",
        return_ads_components: bool = False,
        return_si_eta: bool = False,
    ) -> torch.Tensor:
        """Predict x_1 (absolute coordinates) directly.

        Args:
            pos:        (B, N, 3)  x_0 (structured prior sample, static anchor).
            x_t:        (B, N, 3)  current interpolated state = (1-t)*x_0 + t*x_1.
            t:          (B,)       time in (0, 1).
            atomic_numbers, tags, movable_mask, pad_mask, cell: per-atom features.
            ads_ref_pos: (B, N, 3) optional centered reference geometry on
                tag==2 atoms (zeros elsewhere). Required when
                ``cfg.use_ads_ref_pos`` is True.

        Returns:
            pred_x_1:   (B, N, 3)  predicted final positions. Non-movable atoms
                        held at pos; padding zeroed.
        """
        if not torch.isfinite(pos).all():
            raise RuntimeError("Non-finite values in pos")
        if not torch.isfinite(x_t).all():
            raise RuntimeError("Non-finite values in x_t")

        # 1. Input embeddings
        tokens = self._encode_tokens(
            pos=pos, x_t=x_t, atomic_numbers=atomic_numbers,
            tags=tags, movable_mask=movable_mask, pad_mask=pad_mask,
            cell=cell, ads_ref_pos=ads_ref_pos,
            ads_center_t=ads_center_t, ads_rel_pos_t=ads_rel_pos_t,
        )  # (B, N, atom_s)
        pair_pos = x_t if self.cfg.use_dynamic_pair_dist else pos
        pair_feats = self._build_pair_features(
            pair_pos, atomic_numbers, tags, pad_mask, cell, ads_ref_pos=ads_ref_pos,
        )  # (B, N, N, atom_z)
        rope_diff_mic = _pair_diff_mic(pair_pos, cell) if self.cfg.use_pair_rope else None

        # 1b. Atom-level single→pair enrichment (AtomMOF-style c_to_p_trans_q/k).
        # Uses full tokens (which include x_t via xt_proj), so pair indirectly
        # sees current state. Masked to the same non-bulk-pair block as pair_feats.
        pad_pair = pad_mask.unsqueeze(2) & pad_mask.unsqueeze(1)
        non_bulk = (tags >= 1)
        enrich_v = (pad_pair
                    & non_bulk.unsqueeze(2)
                    & non_bulk.unsqueeze(1)).to(pair_feats.dtype).unsqueeze(-1)
        enrich = (
            self.pair_from_single_i(tokens).unsqueeze(2)
            + self.pair_from_single_j(tokens).unsqueeze(1)
        ) * enrich_v
        pair_feats = pair_feats + enrich

        # 2. Build condition vector at atom_s dim (time only; cell is per-token)
        c = self.t_embedder(t)

        # 3. Encoder
        x = self.encoder(tokens, c, pad_mask, pair_feats, rope_diff_mic)  # (B, N, atom_s)

        # 4. Atom -> Token
        token_single = self.atom_to_token(x)  # (B, N, token_s)
        token_pair = self.atom_to_token_pair(pair_feats)  # (B, N, N, token_z)
        c_trunk = self.cond_proj(c)  # (B, token_s)

        # 5. Trunk initialization (outer-product pair enrichment)
        s_init = self.trunk_s_init(token_single)  # (B, N, token_s)
        if self.cfg.skip_trunk_enrich:
            z_init = token_pair
        else:
            z_init = token_pair + self.trunk_z_init_1(s_init).unsqueeze(2) + self.trunk_z_init_2(s_init).unsqueeze(1)
        s = self.trunk_s_mlp(s_init)  # (B, N, token_s)
        z = self.trunk_z_mlp(z_init)  # (B, N, N, token_z)

        # 6. Trunk
        x_trunk = self.trunk(s, c_trunk, pad_mask, z, rope_diff_mic)  # (B, N, token_s)

        # 7. Token -> Atom (residual)
        x = x + self.token_to_atom(x_trunk)  # (B, N, atom_s)

        # 8. Decoder (optionally with atom-level pair bias)
        if self.cfg.dec_pair_bias:
            x = self.decoder(x, c, pad_mask, pair_feats, rope_diff_mic)  # (B, N, atom_s)
        else:
            x = self.decoder(x, c, pad_mask, None, rope_diff_mic)  # (B, N, atom_s)

        # 9. Output: direct x_1 prediction (AtomMOF-style, zero-init head).
        # Non-movable atoms are forced to pos_0 (their x_1 equals x_0 by
        # construction). Padding zeroed.
        x = self.out_norm(x)
        out = self.coord_out(x)  # normalized coordinate space
        if self.cfg.use_ads_specific_head and not self.cfg.use_ads_center_rel_head:
            ads_mask = (tags == 2).unsqueeze(-1).to(out.dtype)
            ads_out = self.ads_coord_out(x)
            out = ads_out * ads_mask + out * (1.0 - ads_mask)
        out = self._denormalize_coord(out)

        if self.cfg.use_ads_center_rel_head:
            ads_mask_bool = (tags == 2) & movable_mask & pad_mask
            ads_mask = ads_mask_bool.unsqueeze(-1).to(out.dtype)
            ads_global = self._masked_mean(x, ads_mask_bool, dim=1)
            ads_center = self._denormalize_ads_center(self.ads_center_out(ads_global))
            ads_rel = self._denormalize_ads_rel_pos(self.ads_rel_pos_out(x))
            ads_rel = ads_rel * ads_mask
            if self.cfg.ads_rel_output_sum_zero:
                ads_rel = ads_rel - self._masked_mean(ads_rel, ads_mask_bool, dim=1)[:, None, :] * ads_mask
            ads_out = ads_center[:, None, :] + ads_rel
            out = ads_out * ads_mask + out * (1.0 - ads_mask)

        movable_f = movable_mask.unsqueeze(-1).to(out.dtype)
        if self.cfg.use_langevin_param and mlip_force is None:
            raise ValueError("use_langevin_param=True requires mlip_force")
        if self.cfg.use_langevin_param and mlip_force is not None:
            if str(self.cfg.langevin_eval_on) != "x_t":
                raise ValueError(
                    "DiTDenoiser currently supports Langevin parametrization "
                    "only with langevin_eval_on='x_t'"
                )
            f = mlip_force.detach().to(device=out.device, dtype=out.dtype)
            clip = float(self.cfg.langevin_force_clip)
            if clip > 0.0:
                f = f.clamp(min=-clip, max=clip)
            mode = str(self.cfg.langevin_scale_mode)
            if mode == "global":
                lam = self.langevin_scale(c).to(out.dtype).view(out.shape[0], 1, 1)
            elif mode in {"atom", "vector"}:
                lam = self.langevin_scale(x).to(out.dtype)
                if self.cfg.langevin_use_ads_specific_head:
                    ads_mask = (tags == 2).unsqueeze(-1).to(out.dtype)
                    ads_lam = self.ads_langevin_scale(x).to(out.dtype)
                    lam = ads_lam * ads_mask + lam * (1.0 - ads_mask)
            else:
                raise ValueError(f"Unknown langevin_scale_mode={mode!r}")
            if str(langevin_prediction_type) == "x1":
                lp_factor = (1.0 - t).to(out.dtype).view(out.shape[0], 1, 1)
            elif str(langevin_prediction_type) == "v":
                lp_factor = torch.ones((out.shape[0], 1, 1), device=out.device, dtype=out.dtype)
            else:
                raise ValueError(f"Unknown langevin_prediction_type={langevin_prediction_type!r}")
            out = out + lp_factor * lam * f * movable_f
        pad_f = pad_mask.unsqueeze(-1).to(out.dtype)
        pred_x_1 = out * movable_f + pos * (1 - movable_f)
        pred_x_1 = pred_x_1 * pad_f

        if self.training and not torch.isfinite(pred_x_1).all():
            raise RuntimeError("NaN detected in DiTDenoiser forward")
        if self.cfg.use_si_denoiser or return_si_eta:
            eta = self.eta_out(x)
            if self.cfg.si_denoiser_use_ads_specific_head:
                ads_mask = (tags == 2).unsqueeze(-1).to(eta.dtype)
                eta_ads = self.ads_eta_out(x)
                eta = eta_ads * ads_mask + eta * (1.0 - ads_mask)
            eta = eta * movable_f * pad_f
            if self.training and not torch.isfinite(eta).all():
                raise RuntimeError("NaN detected in DiTDenoiser eta forward")
            out_dict = {"pred_x1": pred_x_1, "eta": eta}
            if return_ads_components and self.cfg.use_ads_center_rel_head:
                out_dict.update({"ads_center": ads_center, "ads_rel_pos": ads_rel})
            return out_dict
        if return_ads_components and self.cfg.use_ads_center_rel_head:
            return {
                "pred_x1": pred_x_1,
                "ads_center": ads_center,
                "ads_rel_pos": ads_rel,
            }
        return pred_x_1
