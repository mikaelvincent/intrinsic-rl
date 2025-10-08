"""Configuration package.

This module provides:
- Typed dataclass schemas for the project configuration.
- YAML loaders that parse user configs into those dataclasses.
- Basic validation helpers for common invariants.

Typical usage:
    from irl.cfg import load_config, Config
    cfg: Config = load_config("configs/bipedal_proposed.yaml")
"""

from __future__ import annotations

from .schema import (
    Config,
    EnvConfig,
    PPOConfig,
    IntrinsicConfig,
    GateConfig,
    AdaptationConfig,
    EvaluationConfig,
    LoggingConfig,
)
from .loader import (
    ConfigError,
    load_config,
    loads_config,
    validate_config,
    to_dict,
)

__all__ = [
    # Schemas
    "Config",
    "EnvConfig",
    "PPOConfig",
    "IntrinsicConfig",
    "GateConfig",
    "AdaptationConfig",
    "EvaluationConfig",
    "LoggingConfig",
    # API
    "ConfigError",
    "load_config",
    "loads_config",
    "validate_config",
    "to_dict",
]
