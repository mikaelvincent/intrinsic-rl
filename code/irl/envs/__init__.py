"""Environment manager and wrappers (frame-skip, discrete CarRacing, domain rand).

Import `EnvManager` to build single or vectorized Gymnasium environments with
consistent seeding and shared wrapper logic.
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
