"""Architecture variant registry for AdsorbGen.

Each entry is a dict of ``DiTDenoiserV2Config`` field overrides that defines
one architecture ablation. The baseline (``v2``) leaves every field at its
default; named variants strip or swap a single component so the search space
is compositional and attributable.

Usage from the CLI:

    python -m adsorbgen.train --variant v3-no-mic-dist --out runs/v3-no-mic-dist ...

``build_config`` in ``train.py`` merges the variant's overrides on top of the
CLI flags so ``--variant`` wins over flag defaults but still yields to
explicit CLI overrides passed on the same command line.
"""

from __future__ import annotations

from typing import Any

# Each variant toggles exactly one axis vs the baseline so the contribution
# of that axis is attributable from a single A/B comparison.
VARIANTS: dict[str, dict[str, Any]] = {
    # Baseline: all features on, matches current runs/v2.
    "v2": {},
    # ---- pair-feature axis -----------------------------------------------
    "v3-no-mic-dist": {"use_mic_distance": False},
    "v3-no-pair-pos": {"use_pair_position": False},
    "v3-no-ads-pair": {"use_ads_pair": False},
    "v3-no-pair-outer": {"use_pair_outer": False},
    "v3-pair-scope-all": {"pair_scope": "all"},

    # ---- distance kernel axis --------------------------------------------
    "v4-gaussian-dist": {"dist_kernel": "gaussian", "dist_rbf_num": 16, "dist_rbf_cutoff": 6.0},
    "v4-gaussian-dist-32": {"dist_kernel": "gaussian", "dist_rbf_num": 32, "dist_rbf_cutoff": 8.0},

    # ---- token-feature axis ----------------------------------------------
    "v5-no-tag": {"use_tag_embed": False},
    "v5-no-movable": {"use_movable_embed": False},
    "v5-no-cell": {"use_cell_embed": False},

    # ---- capacity axis (cheap sanity checks) -----------------------------
    "v6-depth-8": {"depth": 8},
    "v6-depth-20": {"depth": 20},
    "v6-dim-768": {"dim": 768, "pair_dim": 192, "num_heads": 12},

    # ---- catalyst-specific axes ------------------------------------------
    # Slab physics: only wrap in-plane (a,b); c has vacuum so no z wrap.
    "v7-2d-pbc": {"pair_pbc": "2d"},
    # Explicit ads-surface interaction channel (symmetric mask).
    "v8-ads-surf-pair": {"use_ads_surf_pair": True},

    # ---- dynamic pair axis ------------------------------------------------
    # Pair distances from x_t (noisy intermediate) instead of static pos.
    "v9-dynamic-pair": {"use_dynamic_pair_dist": True},

    # ---- self-conditioning axis ------------------------------------------
    # AF3/Chen+22 self-conditioning: feed the previous delta_1 prediction in
    # as an extra token-feature via a zero-init Linear(3, dim). Training uses
    # a 2-pass 50% trick (train.py); inference threads the last Euler step's
    # detached prediction through prev_pred.
    "v10-self-cond": {"use_self_cond": True},

    # ---- cross-attention two-stream axis ---------------------------------
    # Replace DiT stack with DiTCrossAttn: per block, ads tokens do self-attn
    # among ads, then cross-attn into surf keys (tag==1), then FFN. Surface
    # tokens pass through unchanged as static context. Blocks are ~1.5x the
    # params of a baseline DiT block, so depth may need tuning.
    "v11-cross-attn": {"use_cross_attn": True},

    # ---- v1 ablation axis ---------------------------------------------------
    # v1 without outer-product pair enrichment at encoder→trunk transition.
    # Requires --arch v1. Tests whether the enrichment is key to v1's edge.
    "v1-no-enrich": {"skip_trunk_enrich": True},
    # v1 + AttentionPairBias on the decoder (atom-level pair features).
    # Isolates "decoder sees pair geometry" axis of v1→v2. Requires --arch v1.
    "v1-dec-pair": {"dec_pair_bias": True},
    # (removed: v1-dec-pair-no-de — ΔE conditioning deleted from codebase.)
    # v1 topology (enc/trunk/dec hierarchy) but with uniform width: enc/dec
    # match trunk (atom_s=token_s=512, atom_z=token_z=256). Isolates the
    # "width" axis of v1→v2: if this matches v2/v6-depth-20, the jump was
    # driven by widening enc/dec, not by flattening the hierarchy.
    # Requires --arch v1.
    "v1-wide": {"atom_s": 512, "atom_z": 256},
    # v1-wide + replace atom_to_token / atom_to_token_pair / token_to_atom
    # Linear+ReLU bridges with Identity. Widths already match in v1-wide, so
    # the bridges are pure gates. Isolates the "inter-stage ReLU gating" axis
    # of v1-wide→v2. Requires --arch v1.
    "v1-wide-no-gate": {"atom_s": 512, "atom_z": 256, "skip_stage_gates": True},
    # Production v1-wide-no-gate stacked with two v2-derived features:
    # (1) decoder pair bias (narrow v1 evidence: +15%p strict valid),
    # (2) Gaussian RBF distance kernel (v4-gaussian-dist: +2.6%p over v2).
    # Used for the 'first_trial_RL' abs-coord flow matching run.
    "v1-wide-no-gate-plus": {
        "atom_s": 512, "atom_z": 256, "skip_stage_gates": True,
        "dec_pair_bias": True, "dist_kernel": "gaussian",
        "dist_rbf_num": 16, "dist_rbf_cutoff": 6.0,
    },
    # v0: post-AtomMOF-style-refactor baseline (absolute-coord flow matching
    # with direct x_1 output, atom-level single→pair enrichment). Inherits the
    # v1-wide-no-gate-plus capacity + pair feature choices; the architectural
    # change (forward signature + output head semantics + enrichment layers)
    # lives in model.py / flow.py, not in the variant config.
    "v0": {
        "atom_s": 512, "atom_z": 256, "skip_stage_gates": True,
        "dec_pair_bias": True, "dist_kernel": "gaussian",
        "dist_rbf_num": 16, "dist_rbf_cutoff": 6.0,
    },
    # v0 + CatFlow-style adsorbate reference geometry embedding. Each tag==2
    # atom additionally receives a Linear(3, atom_s) projection of the
    # centered canonical molecule pose, telling the model the bond pattern
    # the molecule should preserve. Built to attack the high dissoc rate.
    "v0-ads-ref": {
        "atom_s": 512, "atom_z": 256, "skip_stage_gates": True,
        "dec_pair_bias": True, "dist_kernel": "gaussian",
        "dist_rbf_num": 16, "dist_rbf_cutoff": 6.0,
        "use_ads_ref_pos": True,
    },
    # v0-ads-ref + branch-wise typed pair conditioning inside the DiT pair
    # bias. The denoiser stays DiT; branches are fused as a residual pair
    # feature over ads-ads, ads-bond, ads-surface, surface-surface, and
    # surface-bulk relations. Intended to combine with the all-pair auxiliary
    # loss without destabilizing coordinates like the pure typed GNN attempt.
    "v0-ads-ref-branchpair": {
        "atom_s": 512, "atom_z": 256, "skip_stage_gates": True,
        "dec_pair_bias": True, "dist_kernel": "gaussian",
        "dist_rbf_num": 16, "dist_rbf_cutoff": 6.0,
        "use_ads_ref_pos": True,
        "use_typed_pair_features": True,
        "typed_pair_include_bulk": True,
        "max_topological_distance": 8,
        "ads_bond_factor": 1.25,
    },
    # v0-ads-ref + separate final coordinate projection for adsorbate atoms.
    # This mirrors CatFlow's split output heads at the smallest possible
    # AdsorbGen diff: the transformer decoder is still shared, but tag==2
    # atoms no longer share the final coordinate basis with surface atoms.
    "v0-ads-ref-adshead": {
        "atom_s": 512, "atom_z": 256, "skip_stage_gates": True,
        "dec_pair_bias": True, "dist_kernel": "gaussian",
        "dist_rbf_num": 16, "dist_rbf_cutoff": 6.0,
        "use_ads_ref_pos": True,
        "use_ads_specific_head": True,
    },
    # ~2x parameter-count version of v0-ads-ref-adshead. Width is scaled
    # 512→640 and pair width 256→320 while preserving 128-dim attention heads
    # (4/8/4 heads → 5/10/5). Trunk depth 16→22 brings the total to ~206M
    # parameters, i.e. ~2.02x the 101.9M base.
    "v0-ads-ref-adshead-2x": {
        "atom_s": 640, "atom_z": 320,
        "token_s": 640, "token_z": 320,
        "enc_depth": 2, "trunk_depth": 22, "dec_depth": 2,
        "enc_heads": 5, "trunk_heads": 10, "dec_heads": 5,
        "skip_stage_gates": True,
        "dec_pair_bias": True, "dist_kernel": "gaussian",
        "dist_rbf_num": 16, "dist_rbf_cutoff": 6.0,
        "use_ads_ref_pos": True,
        "use_ads_specific_head": True,
    },
    # v0-ads-ref-adshead-2x + lightweight fixed preconditioning.
    # Only x_t (input to xt_proj) and the model's coord output are scaled by
    # coord_scale; pos (x_0) and ads_ref_pos stay raw Å. coord_scale=4.0 is
    # roughly Å→nm with a softer 4 Å unit so adsorbate sub-Å detail is not
    # squashed.
    "v0-ads-ref-adshead-2x-fixedscale": {
        "atom_s": 640, "atom_z": 320,
        "token_s": 640, "token_z": 320,
        "enc_depth": 2, "trunk_depth": 22, "dec_depth": 2,
        "enc_heads": 5, "trunk_heads": 10, "dec_heads": 5,
        "skip_stage_gates": True,
        "dec_pair_bias": True, "dist_kernel": "gaussian",
        "dist_rbf_num": 16, "dist_rbf_cutoff": 6.0,
        "use_ads_ref_pos": True,
        "use_ads_specific_head": True,
        "coord_norm_mode": "fixedscale",
        "coord_mean": (0.0, 0.0, 0.0),
        "coord_scale": (4.0, 4.0, 4.0),
        # ads_ref / pair stay raw — these fields are unused by the model and
        # are kept identity to make args.json reflect the actual behavior.
        "ads_ref_mean": (0.0, 0.0, 0.0),
        "ads_ref_scale": (1.0, 1.0, 1.0),
    },
    # v0-ads-ref-adshead-2x + train-set coordinate statistics for x_t/output.
    # coord_scale uses x1 all-atom std from is2res_train_unwrap_centered.
    # pos (x_0) and ads_ref_pos stay raw Å; ads_ref_* fields are unused.
    "v0-ads-ref-adshead-2x-statnorm": {
        "atom_s": 640, "atom_z": 320,
        "token_s": 640, "token_z": 320,
        "enc_depth": 2, "trunk_depth": 22, "dec_depth": 2,
        "enc_heads": 5, "trunk_heads": 10, "dec_heads": 5,
        "skip_stage_gates": True,
        "dec_pair_bias": True, "dist_kernel": "gaussian",
        "dist_rbf_num": 16, "dist_rbf_cutoff": 6.0,
        "use_ads_ref_pos": True,
        "use_ads_specific_head": True,
        "coord_norm_mode": "statnorm",
        "coord_mean": (0.0, 0.0, 0.0),
        "coord_scale": (3.57, 4.09, 3.71),
        # ads_ref / pair stay raw — fields kept identity for clarity.
        "ads_ref_mean": (0.0, 0.0, 0.0),
        "ads_ref_scale": (1.0, 1.0, 1.0),
    },
    # Same H200 statnorm backbone, but CatFlow-style adsorbate coordinate
    # factorization: decoder predicts ads center from pooled ads tokens and
    # ads rel-pos from per-ads-token features, then assembles center + rel.
    "v0-ads-ref-2x-statnorm-catflow-center-rel": {
        "atom_s": 640, "atom_z": 320,
        "token_s": 640, "token_z": 320,
        "enc_depth": 2, "trunk_depth": 22, "dec_depth": 2,
        "enc_heads": 5, "trunk_heads": 10, "dec_heads": 5,
        "skip_stage_gates": True,
        "dec_pair_bias": True, "dist_kernel": "gaussian",
        "dist_rbf_num": 16, "dist_rbf_cutoff": 6.0,
        "use_ads_ref_pos": True,
        "use_ads_specific_head": False,
        "use_ads_center_rel_head": True,
        "coord_norm_mode": "statnorm",
        "coord_mean": (0.0, 0.0, 0.0),
        "coord_scale": (3.57, 4.09, 3.71),
        "ads_ref_mean": (0.0, 0.0, 0.0),
        "ads_ref_scale": (1.0, 1.0, 1.0),
        "ads_center_mean": (-0.4249, -0.0863, 5.8592),
        "ads_center_scale": (2.9450, 3.2665, 1.7597),
        "ads_rel_pos_mean": (0.0, 0.0, 0.0),
        "ads_rel_pos_scale": (0.6319, 0.8760, 0.7194),
        # CatFlow enforces sum-to-zero in the rel-pos prior; the decoder head
        # itself is trained toward mean-zero rel-pos rather than hard-projected.
        "ads_rel_output_sum_zero": False,
    },
    # v0-ads-ref + dynamic pair distance from x_t (the noisy intermediate),
    # WITH the original MIC convention applied to pair displacements.
    # Tests whether letting pair features see the current sample state
    # improves over the static x_0-only pair channel.
    "v0-ads-ref-dynpair-mic": {
        "atom_s": 512, "atom_z": 256, "skip_stage_gates": True,
        "dec_pair_bias": True, "dist_kernel": "gaussian",
        "dist_rbf_num": 16, "dist_rbf_cutoff": 6.0,
        "use_ads_ref_pos": True,
        "use_dynamic_pair_dist": True, "pair_use_mic": True,
    },
    # v0-ads-ref + dynamic pair distance from x_t WITHOUT MIC. Pair displacements
    # are raw cartesian. Compared against v0-ads-ref-dynpair-mic this isolates
    # the MIC contribution under dynamic pair geometry.
    "v0-ads-ref-dynpair-nomic": {
        "atom_s": 512, "atom_z": 256, "skip_stage_gates": True,
        "dec_pair_bias": True, "dist_kernel": "gaussian",
        "dist_rbf_num": 16, "dist_rbf_cutoff": 6.0,
        "use_ads_ref_pos": True,
        "use_dynamic_pair_dist": True, "pair_use_mic": False,
    },

    # ---- arch comparison -------------------------------------------------
    # Retrain v1 (encoder-trunk-decoder) under v2's data protocol for an
    # apples-to-apples comparison. Requires ``--arch v1`` on the CLI; the
    # empty override dict is a placeholder so search_rank lists the row.
    "v1-retrained": {},
}


def get_variant(name: str) -> dict[str, Any]:
    """Return the override dict for a named variant (empty for baseline)."""
    if name not in VARIANTS:
        raise KeyError(
            f"Unknown variant {name!r}. Known variants: {sorted(VARIANTS.keys())}"
        )
    return dict(VARIANTS[name])


def list_variants() -> list[str]:
    return sorted(VARIANTS.keys())
