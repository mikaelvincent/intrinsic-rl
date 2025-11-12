"""Env manager and small wrappers (frame-skip, CarRacing discrete, domain rand).

Import ``EnvManager`` to build single or vectorized Gymnasium environments
with consistent seeding.
"""

from __future__ import annotations

from .manager import EnvManager
from .wrappers import (
    FrameSkip,
    CarRacingDiscreteActionWrapper,
    DomainRandomizationWrapper,
)

__all__ = [
    "EnvManager",
    "FrameSkip",
    "CarRacingDiscreteActionWrapper",
    "DomainRandomizationWrapper",
]
