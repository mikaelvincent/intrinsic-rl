"""Intrinsic reward modules (skeleton + implemented baselines).

Concrete modules:
- ICM (implemented)
- RND (implemented)
- RIDE (impact-only; reuses ICM encoder; episodic binning arrives next)
- R-IAC (LP via per-region EMAs; implemented Sprint 3 Step 2)
- Proposed unified method (stub)

Base protocol and simple output container are provided here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


# Public base API
class Transition(Protocol):  # minimal protocol for type hints
    s: Any
    a: Any
    r_ext: float
    s_next: Any


@dataclass
class IntrinsicOutput:
    """Represents intrinsic reward computation result for one transition."""

    r_int: float


class BaseIntrinsicModule:
    """Base class for all intrinsic modules."""

    def compute(
        self, tr: Transition
    ) -> IntrinsicOutput:  # pragma: no cover - to be implemented by modules
        """Compute intrinsic reward contribution for a single transition."""
        raise NotImplementedError


# Factory & helpers to integrate modules into the trainer
from .factory import (  # noqa: E402
    is_intrinsic_method,
    create_intrinsic_module,
    compute_intrinsic_batch,
    update_module,
)

# Normalization utility (global intrinsic scaler)
from .normalization import RunningRMS  # noqa: E402

# Export handy concrete classes where useful
from .ride import RIDE  # noqa: E402
from .riac import RIAC  # noqa: E402

__all__ = [
    # Base protocol & output
    "Transition",
    "IntrinsicOutput",
    "BaseIntrinsicModule",
    # Factory helpers
    "is_intrinsic_method",
    "create_intrinsic_module",
    "compute_intrinsic_batch",
    "update_module",
    # Normalization utility
    "RunningRMS",
    # Concrete modules
    "RIDE",
    "RIAC",
]
