from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Mapping

from irl.cfg.schema import GateConfig, IntrinsicConfig
from irl.methods.spec import is_glpe_family, normalize_method


def _get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_bool(v: Any, default: bool) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return bool(v)


def _as_int(v: Any, default: int) -> int:
    if v is None:
        return int(default)
    if isinstance(v, bool):
        raise ValueError("Expected int-like value, got bool")
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    if isinstance(v, str):
        return int(v.strip())
    return int(v)


def _as_float(v: Any, default: float) -> float:
    if v is None:
        return float(default)
    if isinstance(v, bool):
        raise ValueError("Expected float-like value, got bool")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return float(v.strip())
    return float(v)


def build_intrinsic_kwargs(cfg_like: Any) -> dict[str, Any]:
    cfg = cfg_like if is_dataclass(cfg_like) else cfg_like

    method = normalize_method(_get(cfg, "method", "vanilla"))

    intrinsic_defaults = IntrinsicConfig()
    gate_defaults = GateConfig()

    intrinsic = _get(cfg, "intrinsic", intrinsic_defaults)
    gate = _get(intrinsic, "gate", gate_defaults)

    gating_enabled = _as_bool(_get(gate, "enabled", gate_defaults.enabled), gate_defaults.enabled)
    if method == "glpe_nogate":
        gating_enabled = False

    cache_interval = _as_int(
        _get(gate, "median_cache_interval", gate_defaults.median_cache_interval),
        gate_defaults.median_cache_interval,
    )
    if is_glpe_family(method) and method != "glpe_cache":
        cache_interval = 1

    return {
        "bin_size": _as_float(
            _get(intrinsic, "bin_size", intrinsic_defaults.bin_size),
            intrinsic_defaults.bin_size,
        ),
        "alpha_impact": _as_float(
            _get(intrinsic, "alpha_impact", intrinsic_defaults.alpha_impact),
            intrinsic_defaults.alpha_impact,
        ),
        "alpha_lp": _as_float(
            _get(intrinsic, "alpha_lp", intrinsic_defaults.alpha_lp),
            intrinsic_defaults.alpha_lp,
        ),
        "region_capacity": _as_int(
            _get(intrinsic, "region_capacity", intrinsic_defaults.region_capacity),
            intrinsic_defaults.region_capacity,
        ),
        "depth_max": _as_int(
            _get(intrinsic, "depth_max", intrinsic_defaults.depth_max),
            intrinsic_defaults.depth_max,
        ),
        "ema_beta_long": _as_float(
            _get(intrinsic, "ema_beta_long", intrinsic_defaults.ema_beta_long),
            intrinsic_defaults.ema_beta_long,
        ),
        "ema_beta_short": _as_float(
            _get(intrinsic, "ema_beta_short", intrinsic_defaults.ema_beta_short),
            intrinsic_defaults.ema_beta_short,
        ),
        "gate_tau_lp_mult": _as_float(
            _get(gate, "tau_lp_mult", gate_defaults.tau_lp_mult),
            gate_defaults.tau_lp_mult,
        ),
        "gate_tau_s": _as_float(
            _get(gate, "tau_s", gate_defaults.tau_s),
            gate_defaults.tau_s,
        ),
        "gate_hysteresis_up_mult": _as_float(
            _get(gate, "hysteresis_up_mult", gate_defaults.hysteresis_up_mult),
            gate_defaults.hysteresis_up_mult,
        ),
        "gate_min_consec_to_gate": _as_int(
            _get(gate, "min_consec_to_gate", gate_defaults.min_consec_to_gate),
            gate_defaults.min_consec_to_gate,
        ),
        "gate_min_regions_for_gating": _as_int(
            _get(gate, "min_regions_for_gating", gate_defaults.min_regions_for_gating),
            gate_defaults.min_regions_for_gating,
        ),
        "gate_median_cache_interval": int(cache_interval),
        "normalize_inside": _as_bool(
            _get(intrinsic, "normalize_inside", intrinsic_defaults.normalize_inside),
            intrinsic_defaults.normalize_inside,
        ),
        "gating_enabled": bool(gating_enabled),
        "checkpoint_include_points": _as_bool(
            _get(
                intrinsic,
                "checkpoint_include_points",
                intrinsic_defaults.checkpoint_include_points,
            ),
            intrinsic_defaults.checkpoint_include_points,
        ),
    }
