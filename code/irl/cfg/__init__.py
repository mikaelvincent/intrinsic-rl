"""Typed configuration dataclasses plus YAML helpers.

Provides strongly-typed schemas for training, environments, intrinsic modules,
and logging, along with loader/validator utilities.
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
