"""Intrinsic reward modules (skeleton).

Concrete modules (ICM, RND, RIDE, R-IAC, Proposed) are provided as empty stubs in this package and will be implemented
in later sprints.
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
    """Base class for all intrinsic modules (stub)."""

    def compute(
        self, tr: Transition
    ) -> IntrinsicOutput:  # pragma: no cover - to be implemented later
        """Compute intrinsic reward contribution for a single transition."""
        raise NotImplementedError
