"""YAML → dataclass loader and validators for IRL configs."""

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


# For robust Union checks (supports PEP 604 `X | Y`)
try:  # Python 3.10+
    from types import UnionType as _UnionType  # type: ignore
except Exception:  # pragma: no cover - older Python
    _UnionType = None

_UNION_TYPES = (Union,) + ((_UnionType,) if _UnionType is not None else ())


# ----- Public API ------------------------------------------------------------


def load_config(path: Union[str, "Path"]) -> Config:
    """Load a configuration from a YAML file into a typed dataclass.

    Args:
        path: Path to YAML file.

    Returns:
        Config: fully constructed and validated configuration object.

    Raises:
        ConfigError: On YAML syntax errors or schema violations.
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
    """Load configuration from YAML text."""
    try:
        data = yaml.safe_load(yaml_text) or {}
    except Exception as exc:
        raise ConfigError(f"Invalid YAML: {exc}") from exc

    if not isinstance(data, Mapping):
        raise ConfigError(f"Top-level YAML must be a mapping, got {type(data).__name__}")

    cfg = _from_mapping(Config, data, path="config")
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    """Validate common invariants for PPO training & environment settings.

    Raises:
        ConfigError: if a validation rule is violated.
    """
    # Basic ranges
    if cfg.env.vec_envs < 1:
        raise ConfigError("env.vec_envs must be >= 1")

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

    if cfg.intrinsic.r_clip <= 0.0:
        raise ConfigError("intrinsic.r_clip must be > 0")

    # Batch divisibility rule (supporting both 'total steps' and 'per-env steps' interpretations)
    # Accept if either interpretation yields an integer minibatch size.
    total_transitions_variant_a = cfg.ppo.steps_per_update  # interpreted as global total per update
    total_transitions_variant_b = (
        cfg.ppo.steps_per_update * cfg.env.vec_envs
    )  # if steps are per-env

    if (total_transitions_variant_a % cfg.ppo.minibatches != 0) and (
        total_transitions_variant_b % cfg.ppo.minibatches != 0
    ):
        raise ConfigError(
            "Minibatch divisibility violated: either `ppo.steps_per_update` (interpreted as "
            "global transitions per update) or `ppo.steps_per_update * env.vec_envs` "
            "(interpreted as per-env steps × #envs) must be divisible by `ppo.minibatches`."
        )

    # Method-specific gentle hints (no hard failure)
    if cfg.method == "vanilla":
        # Vanilla PPO ignores intrinsic, but keeping section is harmless.
        pass

    # CarRacing default: discrete actions encouraged by spec
    if cfg.env.id.startswith("CarRacing") and not cfg.env.discrete_actions:
        # No error — just a soft check placeholder. Real policy head checks happen at runtime.
        pass


def to_dict(cfg: Config) -> dict:
    """Convert Config dataclass (recursively) into a plain dict."""
    return asdict(cfg)


# ----- Internal: recursive mapping → dataclass conversion --------------------


def _from_mapping(cls: Type[Any], data: Mapping[str, Any], path: str) -> Any:
    """Recursively coerce a Mapping into the dataclass `cls`.

    Important:
        Works correctly even when dataclasses were defined with
        `from __future__ import annotations` (PEP 563), by resolving field types
        via `typing.get_type_hints`.
    """
    if not is_dataclass(cls):
        raise ConfigError(f"Internal error: target {cls!r} is not a dataclass")

    # Strict unknown field detection
    allowed = {f.name for f in fields(cls)}
    unknown = set(data.keys()) - allowed
    if unknown:
        pretty = ", ".join(sorted(unknown))
        raise ConfigError(f"Unknown field(s) at {path}: {pretty}")

    # Resolve real (non-deferred) field types
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
            # Allow dataclass defaults / factories to apply; error only if neither exists.
            if f.default is not MISSING or f.default_factory is not MISSING:  # type: ignore[attr-defined]
                continue
            raise ConfigError(f"Missing required field: {path}.{key}")

    try:
        return cls(**kwargs)  # type: ignore[misc]
    except Exception as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Failed to construct {cls.__name__} at {path}: {exc}") from exc


def _coerce_value_to_type(value: Any, typ: Any, path: str) -> Any:
    """Best-effort coercion of YAML-loaded values into annotated types."""
    origin = get_origin(typ)
    args = get_args(typ)

    # Dataclass types
    if is_dataclass_type(typ):
        if not isinstance(value, Mapping):
            raise ConfigError(f"Expected mapping at {path}, got {type(value).__name__}")
        return _from_mapping(typ, value, path)

    # Optional / Union (supports typing.Union and PEP 604 unions)
    if origin in _UNION_TYPES:
        # Optional[T] case
        if type(None) in args:
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

    # Sequences (Lists/Tuples) — accept generic sequences but not strings
    if origin in (list, tuple, ABCSequence):
        elem_type = args[0] if args else Any
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ConfigError(f"Expected sequence at {path}, got {type(value).__name__}")
        return [_coerce_value_to_type(v, elem_type, f"{path}[{i}]") for i, v in enumerate(value)]

    # Mappings (Dict-like) — shallow accept; keys/values left as-is
    if origin in (dict, ABCMapping):
        if not isinstance(value, Mapping):
            raise ConfigError(f"Expected mapping at {path}, got {type(value).__name__}")
        return dict(value)

    # Primitive coercions
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
            # Prevent bool subclasses from slipping through as ints
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

    # Any / unknown typing — accept as-is
    return value


def is_dataclass_type(t: Any) -> bool:
    try:
        return is_dataclass(t)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - defensive
        return False
