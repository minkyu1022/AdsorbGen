"""Inference CLI and checkpoint-loading helpers."""

from adsorbgen.inference.cli import (  # noqa: F401
    _extract_state_dict,
    _filter_dataclass_fields,
    _make_forward,
    _resolve_model_cfg,
    main,
)
