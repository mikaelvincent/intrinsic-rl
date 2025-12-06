"""Typed configuration dataclasses, loader, and validators."""

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
    ExperimentConfig,
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
    "ExperimentConfig",
    # API
    "ConfigError",
    "load_config",
    "loads_config",
    "validate_config",
    "to_dict",
]
