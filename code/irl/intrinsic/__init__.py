"""Intrinsic reward modules (skeleton + implemented baselines).

Concrete modules:
- ICM (implemented)
- RND (implemented)
- RIDE (stub)
- R-IAC (stub)
- Proposed unified method (stub)

Base protocol and simple output container are provided here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


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
