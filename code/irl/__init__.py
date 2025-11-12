"""Intrinsic RL research codebase for impact and learning-progress methods."""

from ._version import __version__

# Expose utilities for convenience (logging & checkpoints)
from .utils.loggers import CSVLogger, TBLogger, MetricLogger  # noqa: F401
from .utils.checkpoint import CheckpointManager, load_checkpoint  # noqa: F401

__all__ = [
    "__version__",
    "CSVLogger",
    "TBLogger",
    "MetricLogger",
    "CheckpointManager",
    "load_checkpoint",
]
