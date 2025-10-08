"""Intrinsic RL research codebase (skeleton).

This package skeleton is created in Sprint 0 — Step 1. Subpackages are empty scaffolds to be filled in subsequent steps.
"""

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
