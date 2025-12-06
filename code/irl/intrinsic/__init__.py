"""Intrinsic modules and base protocol.

Provides base interfaces and concrete modules (ICM, RND, RIDE, RIAC, Proposed)
plus factory helpers and a running RMS normalizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


# Public base API
class Transition(Protocol):
    """Minimal transition protocol shared by intrinsic modules.

    Attributes
    ----------
    s : Any
        Observation at time :math:`t`.
    a : Any
        Action taken at time :math:`t`.
    r_ext : float
        Extrinsic reward observed after taking the action.
    s_next : Any
        Observation at time :math:`t + 1`.
    """

    s: Any
    a: Any
    r_ext: float
    s_next: Any


@dataclass
class IntrinsicOutput:
    """Result of computing intrinsic reward for one transition.

    Attributes
    ----------
    r_int : float
        Intrinsic reward contribution for the transition.
    """

    r_int: float


class BaseIntrinsicModule:
    """Abstract base class for intrinsic reward modules.

    Subclasses implement :meth:`compute`, which maps a single transition
    to an :class:`IntrinsicOutput`. Batch-oriented variants are provided
    by helpers in :mod:`irl.intrinsic.factory`.
    """

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
from .proposed import Proposed  # noqa: E402

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
    "Proposed",
]
