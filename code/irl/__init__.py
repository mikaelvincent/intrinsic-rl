"""Intrinsic RL research codebase for impact and learning-progress methods."""

import os

# Disable NNPACK to prevent "Unsupported hardware" crashes/hangs on some
# containerized CPUs during CNN evaluation (e.g. CarRacing).
os.environ.setdefault("ATEN_NNPACK_ENABLED", "0")

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
