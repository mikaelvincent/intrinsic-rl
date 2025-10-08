"""Environment managers and wrappers.

Vectorized environment creation, seeding, and simple domain randomization wrappers.

Usage:
    from irl.envs import EnvManager

This module provides:
- EnvManager: builds single or vectorized Gymnasium environments with consistent seeding.
- Lightweight wrappers: FrameSkip, CarRacingDiscreteActionWrapper, DomainRandomizationWrapper.

The full training loop will integrate this manager in later sprints.
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
