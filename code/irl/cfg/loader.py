from __future__ import annotations

import sys
import warnings
from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
from dataclasses import MISSING, asdict, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

from .schema import Config


class ConfigError(ValueError):
    pass


try:
    from types import UnionType as _UnionType
except Exception:
    _UnionType = None

_UNION_TYPES = (Union,) + ((_UnionType,) if _UnionType is not None else ())


def load_config(path: Union[str, Path]) -> Config:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        raise ConfigError(f"Failed to read config file: {p}") from exc
    return loads_config(text)


def loads_config(yaml_text: str) -> Config:
    try:
        data = yaml.safe_load(yaml_text) or {}
    except Exception as exc:
        raise ConfigError(f"Invalid configuration data: {exc}") from exc

    if not isinstance(data, Mapping):
        raise ConfigError(f"Top-level configuration must be a mapping, got {type(data).__name__}")

    cfg = _from_mapping(Config, data, path="config")
    validate_config(cfg)
    return cfg


def _divisors(n: int) -> list[int]:
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
    seed_val = getattr(cfg, "seed", 1)
    if isinstance(seed_val, bool):
        raise ConfigError("seed must be an int or list of ints")
    if isinstance(seed_val, (list, tuple)):
        if not seed_val:
            raise ConfigError("seed must be a non-empty int or list of ints")
        for i, s in enumerate(seed_val):
            if isinstance(s, bool):
                raise ConfigError(f"seed[{i}] must be an int, got bool")
            try:
                _ = int(s)
            except Exception as exc:
                raise ConfigError(f"seed[{i}] must be int-like, got {s!r}") from exc

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
    if cfg.ppo.value_coef <= 0.0:
        raise ConfigError("ppo.value_coef must be > 0")
    if cfg.ppo.value_clip_range < 0.0:
        raise ConfigError("ppo.value_clip_range must be >= 0 (0 disables value clipping)")
    if cfg.ppo.kl_penalty_coef < 0.0:
        raise ConfigError("ppo.kl_penalty_coef must be >= 0")
    if cfg.ppo.kl_stop < 0.0:
        raise ConfigError("ppo.kl_stop must be >= 0 (0 disables early stop)")
    if cfg.intrinsic.r_clip <= 0.0:
        raise ConfigError("intrinsic.r_clip must be > 0")

    method = str(cfg.method).lower()
    method_base = "glpe" if method.startswith("glpe_") else method

    is_glpe_family = method.startswith("glpe")
    start = getattr(cfg.intrinsic, "taper_start_frac", None)
    end = getattr(cfg.intrinsic, "taper_end_frac", None)

    if (start is not None) or (end is not None):
        if not is_glpe_family:
            raise ConfigError(
                "`intrinsic.taper_start_frac` and `intrinsic.taper_end_frac` apply only to glpe* methods."
            )

        if (start is None) or (end is None):
            raise ConfigError(
                "Set both `intrinsic.taper_start_frac` and `intrinsic.taper_end_frac`, or omit both."
            )

        if isinstance(start, bool) or isinstance(end, bool):
            raise ConfigError(
                "`intrinsic.taper_start_frac` and `intrinsic.taper_end_frac` must be float values."
            )

        try:
            s = float(start)
            e = float(end)
        except Exception as exc:
            raise ConfigError(
                "`intrinsic.taper_start_frac` and `intrinsic.taper_end_frac` must be float values."
            ) from exc

        if not (0.0 <= s < e <= 1.0):
            raise ConfigError(
                "`intrinsic.taper_start_frac` and `intrinsic.taper_end_frac` must satisfy "
                f"0.0 <= start < end <= 1.0; got start={s}, end={e}."
            )

    if method == "glpe_lp_only":
        if cfg.intrinsic.alpha_impact != 0.0:
            raise ConfigError("intrinsic.alpha_impact must be 0 for method 'glpe_lp_only'")
        if cfg.intrinsic.alpha_lp <= 0.0:
            raise ConfigError("intrinsic.alpha_lp must be > 0 for method 'glpe_lp_only'")

    if method == "glpe_impact_only":
        if cfg.intrinsic.alpha_lp != 0.0:
            raise ConfigError("intrinsic.alpha_lp must be 0 for method 'glpe_impact_only'")
        if cfg.intrinsic.alpha_impact <= 0.0:
            raise ConfigError("intrinsic.alpha_impact must be > 0 for method 'glpe_impact_only'")

    if method == "glpe_nogate":
        if bool(cfg.intrinsic.gate.enabled):
            raise ConfigError("intrinsic.gate.enabled must be False for method 'glpe_nogate'")

    if method_base == "ride":
        if cfg.intrinsic.alpha_impact <= 0.0:
            raise ConfigError("intrinsic.alpha_impact must be > 0 for method 'ride'")
    elif method_base == "glpe":
        if cfg.intrinsic.alpha_impact < 0.0:
            raise ConfigError("intrinsic.alpha_impact must be >= 0 for method 'glpe'")
    else:
        if cfg.intrinsic.alpha_impact <= 0.0:
            warnings.warn(
                f"intrinsic.alpha_impact<=0 has no effect for method '{method}'; ignoring.",
                UserWarning,
            )

    if method_base != "glpe":
        try:
            if hasattr(cfg.intrinsic, "normalize_inside") and not cfg.intrinsic.normalize_inside:
                warnings.warn(
                    "intrinsic.normalize_inside is only used for method 'glpe'; "
                    f"ignoring for method '{method}'.",
                    UserWarning,
                )
        except Exception:
            pass
        try:
            if hasattr(cfg.intrinsic, "gate") and hasattr(cfg.intrinsic.gate, "enabled"):
                if not bool(cfg.intrinsic.gate.enabled):
                    warnings.warn(
                        "intrinsic.gate.enabled is only used for method 'glpe'; "
                        f"ignoring for method '{method}'.",
                        UserWarning,
                    )
        except Exception:
            pass

    if method_base == "ride":
        if cfg.intrinsic.bin_size <= 0.0:
            raise ConfigError("intrinsic.bin_size must be > 0 for method 'ride'")
    else:
        if cfg.intrinsic.bin_size <= 0.0:
            warnings.warn(
                f"intrinsic.bin_size<=0 is irrelevant for method '{method}'; ignoring.",
                UserWarning,
            )

    if method_base == "glpe":
        gate_enabled = True
        try:
            gate_enabled = bool(cfg.intrinsic.gate.enabled)
        except Exception:
            gate_enabled = True

        if method == "glpe_nogate":
            gate_enabled = False

        if method == "glpe_cache":
            if not gate_enabled:
                raise ConfigError("intrinsic.gate.enabled must be True for method 'glpe_cache'")
            if cfg.intrinsic.gate.median_cache_interval <= 1:
                raise ConfigError(
                    "intrinsic.gate.median_cache_interval must be > 1 for method 'glpe_cache'"
                )
        elif method.startswith("glpe"):
            if int(cfg.intrinsic.gate.median_cache_interval) != 1:
                raise ConfigError(
                    f"intrinsic.gate.median_cache_interval must be 1 for method '{method}'. "
                    "Use method 'glpe_cache' to enable median caching."
                )

        if gate_enabled:
            if cfg.intrinsic.gate.min_consec_to_gate <= 0:
                raise ConfigError("intrinsic.gate.min_consec_to_gate must be > 0 for method 'glpe'")
            if cfg.intrinsic.gate.min_regions_for_gating <= 0:
                raise ConfigError(
                    "intrinsic.gate.min_regions_for_gating must be > 0 for method 'glpe'"
                )
            if cfg.intrinsic.gate.median_cache_interval <= 0:
                raise ConfigError(
                    "intrinsic.gate.median_cache_interval must be >= 1 for method 'glpe'"
                )
        else:
            if method != "glpe_nogate":
                warnings.warn(
                    "GLPE gating disabled via intrinsic.gate.enabled=False; "
                    "gating thresholds will be ignored.",
                    UserWarning,
                )

    total_a = cfg.ppo.steps_per_update
    total_b = cfg.ppo.steps_per_update * cfg.env.vec_envs
    if (total_a % cfg.ppo.minibatches != 0) and (total_b % cfg.ppo.minibatches != 0):
        valid = sorted(set(_divisors(total_a) + _divisors(total_b)))
        try:
            valid.remove(int(cfg.ppo.minibatches))
        except ValueError:
            pass

        lower, higher = _nearest_divisor_suggestions(int(cfg.ppo.minibatches), valid)
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

    try:
        ts = getattr(cfg.exp, "total_steps", None)
        if ts is None:
            raise ConfigError("exp.total_steps is required")
        if int(ts) <= 0:
            raise ConfigError("exp.total_steps must be > 0")
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Invalid exp.total_steps: {exc}") from exc


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)


def _from_mapping(cls: Type[Any], data: Mapping[str, Any], path: str) -> Any:
    if not is_dataclass(cls):
        raise ConfigError(f"Internal error: target {cls!r} is not a dataclass")

    allowed = {f.name for f in fields(cls)}
    unknown = set(data.keys()) - allowed
    if unknown:
        pretty = ", ".join(sorted(unknown))
        raise ConfigError(f"Unknown field(s) at {path}: {pretty}")

    mod = sys.modules.get(cls.__module__)
    gns = mod.__dict__ if mod is not None else None
    try:
        type_hints = get_type_hints(cls, globalns=gns, localns=None)
    except Exception:
        type_hints = {}

    kwargs: MutableMapping[str, Any] = {}
    for f in fields(cls):
        key = f.name
        target_type = type_hints.get(key, f.type)
        if key in data:
            kwargs[key] = _coerce_value_to_type(data[key], target_type, f"{path}.{key}")
        else:
            if f.default is not MISSING or f.default_factory is not MISSING:
                continue
            raise ConfigError(f"Missing required field: {path}.{key}")

    try:
        return cls(**kwargs)
    except Exception as exc:
        raise ConfigError(f"Failed to construct {cls.__name__} at {path}: {exc}") from exc


def _coerce_value_to_type(value: Any, typ: Any, path: str) -> Any:
    origin = get_origin(typ)
    args = get_args(typ)

    if is_dataclass_type(typ):
        if not isinstance(value, Mapping):
            raise ConfigError(f"Expected mapping at {path}, got {type(value).__name__}")
        return _from_mapping(typ, value, path)

    if origin in _UNION_TYPES:
        if type(None) in args:
            if value is None:
                return None
            non_none = [a for a in args if a is not type(None)]
            last_err: Exception | None = None
            for a in non_none:
                try:
                    return _coerce_value_to_type(value, a, path)
                except Exception as exc:
                    last_err = exc
            raise ConfigError(f"Could not coerce value at {path}: {last_err}") from last_err

        for a in args:
            try:
                return _coerce_value_to_type(value, a, path)
            except Exception:
                continue
        raise ConfigError(f"Value at {path!r} does not match any allowed union types.")

    if origin is Literal:
        literals = set(args)
        if value not in literals:
            raise ConfigError(f"{path}: expected one of {sorted(map(repr, literals))}, got {value!r}")
        return value

    if origin in (list, tuple, ABCSequence):
        elem_type = args[0] if args else Any
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ConfigError(f"Expected sequence at {path}, got {type(value).__name__}")
        return [_coerce_value_to_type(v, elem_type, f"{path}[{i}]") for i, v in enumerate(value)]

    if origin in (dict, ABCMapping):
        if not isinstance(value, Mapping):
            raise ConfigError(f"Expected mapping at {path}, got {type(value).__name__}")
        return dict(value)

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
        if isinstance(value, bool):
            raise ConfigError(f"Expected float at {path}, got bool")
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

    return value


def is_dataclass_type(t: Any) -> bool:
    try:
        return is_dataclass(t)
    except Exception:
        return False
