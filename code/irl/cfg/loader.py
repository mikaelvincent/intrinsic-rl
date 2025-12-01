"""Dataclass-based configuration loader and validators for IRL configs.

Provides concise helpers for reading, parsing, and validating configs.
"""

from __future__ import annotations

import sys
import types as _types
from dataclasses import MISSING, asdict, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Mapping,
    MutableMapping,
    Sequence,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    Literal,
)
from collections.abc import Sequence as ABCSequence, Mapping as ABCMapping

import warnings
import yaml  # PyYAML

from .schema import (
    AdaptationConfig,
    Config,
    EnvConfig,
    EvaluationConfig,
    GateConfig,
    IntrinsicConfig,
    LoggingConfig,
    PPOConfig,
)


# ----- Exceptions ------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when configuration parsing or validation fails."""


# Support both typing.Union and PEP 604 unions (X | Y)
try:  # Python 3.10+
    from types import UnionType as _UnionType  # type: ignore
except Exception:  # pragma: no cover - older Python
    _UnionType = None

_UNION_TYPES = (Union,) + ((_UnionType,) if _UnionType is not None else ())


# ----- Public API ------------------------------------------------------------


def load_config(path: Union[str, "Path"]) -> Config:
    """Load and validate a ``Config`` from a configuration file.

    Raises ``ConfigError`` with a descriptive message on read/parse/validation issues.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")

    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - I/O edge
        raise ConfigError(f"Failed to read config file: {p}") from exc

    cfg = loads_config(text)
    return cfg


def loads_config(yaml_text: str) -> Config:
    """Load and validate a ``Config`` from configuration text."""
    try:
        data = yaml.safe_load(yaml_text) or {}
    except Exception as exc:
        raise ConfigError(f"Invalid configuration data: {exc}") from exc

    if not isinstance(data, Mapping):
        raise ConfigError(f"Top-level configuration must be a mapping, got {type(data).__name__}")

    cfg = _from_mapping(Config, data, path="config")
    validate_config(cfg)
    return cfg


# ----- Helpers for friendlier validation messages ---------------------------


def _divisors(n: int) -> list[int]:
    """Return all positive divisors of n (sorted)."""
    n = int(abs(n))
    if n == 0:
        return []
    small, large = [], []
    d = 1
    while d * d <= n:
        if n % d == 0:
            small.append(d)
            if d != n // d:
                large.append(n // d)
        d += 1
    return sorted(small + large[::-1])


def _nearest_divisor_suggestions(
    target: int, candidates: list[int]
) -> tuple[int | None, int | None]:
    """Given a sorted candidate list, return (nearest_lower, nearest_higher) around target (excluding target)."""
    if not candidates:
        return None, None
    lower = None
    higher = None
    for d in candidates:
        if d < target:
            lower = d
        elif d > target and higher is None:
            higher = d
            break
    return lower, higher


def validate_config(cfg: Config) -> None:
    """Check common PPO/env invariants (ranges, divisibility, gentle hints)."""
    # Basic ranges
    if cfg.env.vec_envs < 1:
        raise ConfigError("env.vec_envs must be >= 1")
    if cfg.env.frame_skip < 1:
        raise ConfigError("env.frame_skip must be >= 1")

    if cfg.ppo.steps_per_update <= 0:
        raise ConfigError("ppo.steps_per_update must be > 0")

    if cfg.ppo.minibatches <= 0:
        raise ConfigError("ppo.minibatches must be > 0")

    if not (0.0 < cfg.ppo.gamma <= 1.0):
        raise ConfigError("ppo.gamma must be in (0, 1]")

    if not (0.0 <= cfg.ppo.gae_lambda <= 1.0):
        raise ConfigError("ppo.gae_lambda must be in [0, 1]")

    if cfg.ppo.clip_range <= 0.0:
        raise ConfigError("ppo.clip_range must be > 0")

    # Value loss and clipping checks
    if cfg.ppo.value_coef <= 0.0:
        raise ConfigError("ppo.value_coef must be > 0")
    if cfg.ppo.value_clip_range < 0.0:
        raise ConfigError("ppo.value_clip_range must be >= 0 (0 disables value clipping)")

    # KL controls
    if cfg.ppo.kl_penalty_coef < 0.0:
        raise ConfigError("ppo.kl_penalty_coef must be >= 0")
    if cfg.ppo.kl_stop < 0.0:
        raise ConfigError("ppo.kl_stop must be >= 0 (0 disables early stop)")

    # Always require a sane intrinsic clipping bound (trainer relies on this).
    if cfg.intrinsic.r_clip <= 0.0:
        raise ConfigError("intrinsic.r_clip must be > 0")

    # Method-specific knobs: enforce only when the selected method uses them.
    method = str(cfg.method).lower()

    # alpha_impact rules:
    # - 'ride' requires a strictly positive impact weight
    # - 'proposed' allows a true LP-only ablation (alpha_impact >= 0)
    if method == "ride":
        if cfg.intrinsic.alpha_impact <= 0.0:
            raise ConfigError("intrinsic.alpha_impact must be > 0 for method 'ride'")
    elif method == "proposed":
        if cfg.intrinsic.alpha_impact < 0.0:
            raise ConfigError("intrinsic.alpha_impact must be >= 0 for method 'proposed'")
    else:
        if cfg.intrinsic.alpha_impact <= 0.0:
            warnings.warn(
                f"intrinsic.alpha_impact<=0 has no effect for method '{method}'; ignoring.",
                UserWarning,
            )

    # Proposed-only flags: normalize_inside and gate.enabled
    if method != "proposed":
        try:
            if hasattr(cfg.intrinsic, "normalize_inside") and not cfg.intrinsic.normalize_inside:
                warnings.warn(
                    f"intrinsic.normalize_inside is only used for method 'proposed'; "
                    f"ignoring for method '{method}'.",
                    UserWarning,
                )
        except Exception:
            pass
        try:
            if hasattr(cfg.intrinsic, "gate") and hasattr(cfg.intrinsic.gate, "enabled"):
                if not bool(cfg.intrinsic.gate.enabled):
                    warnings.warn(
                        f"intrinsic.gate.enabled is only used for method 'proposed'; "
                        f"ignoring for method '{method}'.",
                        UserWarning,
                    )
        except Exception:
            pass

    # bin_size controls episodic binning used only by RIDE
    if method == "ride":
        if cfg.intrinsic.bin_size <= 0.0:
            raise ConfigError("intrinsic.bin_size must be > 0 for method 'ride'")
    else:
        if cfg.intrinsic.bin_size <= 0.0:
            warnings.warn(
                f"intrinsic.bin_size<=0 is irrelevant for method '{method}'; ignoring.",
                UserWarning,
            )

    # Gating sanity checks only when Proposed is selected
    if method == "proposed":
        # If gating is explicitly disabled, accept thresholds as provided but do not enforce them.
        gate_enabled = True
        try:
            gate_enabled = bool(cfg.intrinsic.gate.enabled)
        except Exception:
            gate_enabled = True
        if gate_enabled:
            if cfg.intrinsic.gate.min_consec_to_gate <= 0:
                raise ConfigError(
                    "intrinsic.gate.min_consec_to_gate must be > 0 for method 'proposed'"
                )
            if cfg.intrinsic.gate.min_regions_for_gating <= 0:
                raise ConfigError(
                    "intrinsic.gate.min_regions_for_gating must be > 0 for method 'proposed'"
                )
        else:
            # Optional hint to users that gating is off for this run
            warnings.warn(
                "Proposed gating disabled via intrinsic.gate.enabled=False; gating thresholds will be ignored.",
                UserWarning,
            )

    # Minibatch divisibility: accept either interpretation of steps_per_update
    total_a = cfg.ppo.steps_per_update
    total_b = cfg.ppo.steps_per_update * cfg.env.vec_envs
    if (total_a % cfg.ppo.minibatches != 0) and (total_b % cfg.ppo.minibatches != 0):
        # Build actionable suggestions for ppo.minibatches based on divisors of totals.
        divs_a = _divisors(total_a)
        divs_b = _divisors(total_b)
        # Merge, unique, sorted
        valid = sorted(set(divs_a + divs_b))
        # exclude the current (invalid) choice to avoid echoing it back
        try:
            valid.remove(int(cfg.ppo.minibatches))
        except ValueError:
            pass
        lower, higher = _nearest_divisor_suggestions(int(cfg.ppo.minibatches), valid)
        # Trim a tiny showcase of nearby/typical divisors to keep message concise
        showcase = [
            d for d in valid if d in {1, 2, 4, 8, 16, 32, 64, 128, 256} or d in (lower, higher)
        ]
        showcase = sorted(set(showcase))
        hint_bits: list[str] = []
        if lower is not None and higher is not None:
            hint_bits.append(f"try ppo.minibatches={lower} or {higher}")
        elif lower is not None:
            hint_bits.append(f"try ppo.minibatches={lower}")
        elif higher is not None:
            hint_bits.append(f"try ppo.minibatches={higher}")
        if showcase:
            hint_bits.append(f"valid examples: {showcase}")
        hint = " — " + "; ".join(hint_bits) if hint_bits else ""

        raise ConfigError(
            "Minibatch divisibility violated: either `ppo.steps_per_update` or "
            "`ppo.steps_per_update * env.vec_envs` must be divisible by `ppo.minibatches`.\n"
            f"(steps_per_update={total_a}, vec_envs={cfg.env.vec_envs}, "
            f"minibatches={cfg.ppo.minibatches}){hint}"
        )

    # Method-specific hints (soft checks)
    if cfg.method == "vanilla":
        pass

    if cfg.env.id.startswith("CarRacing") and not cfg.env.discrete_actions:
        # Soft guidance; real policy head checks happen at runtime.
        pass


def to_dict(cfg: Config) -> dict:
    """Convert Config dataclass recursively to a plain dict."""
    return asdict(cfg)


# ----- Internal: recursive mapping → dataclass conversion --------------------


def _from_mapping(cls: Type[Any], data: Mapping[str, Any], path: str) -> Any:
    """Recursively coerce a Mapping into the dataclass `cls` (strict)."""
    if not is_dataclass(cls):
        raise ConfigError(f"Internal error: target {cls!r} is not a dataclass")

    # Strict unknown field detection
    allowed = {f.name for f in fields(cls)}
    unknown = set(data.keys()) - allowed
    if unknown:
        pretty = ", ".join(sorted(unknown))
        raise ConfigError(f"Unknown field(s) at {path}: {pretty}")

    # Resolve real field types (robust even with postponed annotations)
    mod = sys.modules.get(cls.__module__)
    gns = mod.__dict__ if mod is not None else None
    try:
        type_hints = get_type_hints(cls, globalns=gns, localns=None)  # robust across Py3.10+
    except Exception:  # pragma: no cover - very defensive
        type_hints = {}

    kwargs: MutableMapping[str, Any] = {}
    for f in fields(cls):
        key = f.name
        target_type = type_hints.get(key, f.type)
        if key in data:
            kwargs[key] = _coerce_value_to_type(data[key], target_type, f"{path}.{key}")
        else:
            if f.default is not MISSING or f.default_factory is not MISSING:  # type: ignore[attr-defined]
                continue
            raise ConfigError(f"Missing required field: {path}.{key}")

    try:
        return cls(**kwargs)  # type: ignore[misc]
    except Exception as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Failed to construct {cls.__name__} at {path}: {exc}") from exc


def _coerce_value_to_type(value: Any, typ: Any, path: str) -> Any:
    """Best-effort coercion of configuration values into annotated types."""
    origin = get_origin(typ)
    args = get_args(typ)

    # Dataclass types
    if is_dataclass_type(typ):
        if not isinstance(value, Mapping):
            raise ConfigError(f"Expected mapping at {path}, got {type(value).__name__}")
        return _from_mapping(typ, value, path)

    # Optional/Union
    if origin in _UNION_TYPES:
        if type(None) in args:  # Optional[T]
            if value is None:
                return None
            non_none = [a for a in args if a is not type(None)]
            last_err: Exception | None = None
            for a in non_none:
                try:
                    return _coerce_value_to_type(value, a, path)
                except Exception as exc:  # pragma: no cover - uncommon
                    last_err = exc
            raise ConfigError(f"Could not coerce value at {path}: {last_err}") from last_err
        # General Union — try each arm
        for a in args:
            try:
                return _coerce_value_to_type(value, a, path)
            except Exception:
                continue
        raise ConfigError(f"Value at {path!r} does not match any allowed union types.")

    # Literals (e.g., method choices)
    if origin is Literal:
        literals = set(args)
        if value not in literals:
            raise ConfigError(
                f"{path}: expected one of {sorted(map(repr, literals))}, got {value!r}"
            )
        return value

    # Sequences (not strings)
    if origin in (list, tuple, ABCSequence):
        elem_type = args[0] if args else Any
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ConfigError(f"Expected sequence at {path}, got {type(value).__name__}")
        return [_coerce_value_to_type(v, elem_type, f"{path}[{i}]") for i, v in enumerate(value)]

    # Mappings
    if origin in (dict, ABCMapping):
        if not isinstance(value, Mapping):
            raise ConfigError(f"Expected mapping at {path}, got {type(value).__name__}")
        return dict(value)

    # Primitives
    if typ is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"1", "true", "yes", "y", "on"}:
                return True
            if low in {"0", "false", "no", "n", "off"}:
                return False
        raise ConfigError(f"Expected bool at {path}, got {value!r}")

    if typ is int:
        if isinstance(value, bool):
            raise ConfigError(f"Expected int at {path}, got bool")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and float(value).is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        raise ConfigError(f"Expected int at {path}, got {value!r}")

    if typ is float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise ConfigError(f"Expected float at {path}, got {value!r}")

    if typ is str:
        if isinstance(value, (str, int, float, bool)):
            return str(value) if not isinstance(value, str) else value
        return str(value)

    # Fallback
    return value


def is_dataclass_type(t: Any) -> bool:
    """Return True if `t` is a dataclass type."""
    try:
        return is_dataclass(t)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - defensive
        return False
