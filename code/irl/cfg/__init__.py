"""Typed config schemas, YAML loader, and validators.

See devspec/dev_spec_and_plan.md §6 for the full schema.
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
