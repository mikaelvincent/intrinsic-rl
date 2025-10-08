"""Model definitions (skeleton).

Policy and value networks and optional encoders will be added in Sprint 0 — Step 5.
"""

from __future__ import annotations

from typing import Any


class PolicyNetwork:  # pragma: no cover - placeholder
    """Stub for the policy network."""

    def __init__(self, obs_space: Any, action_space: Any) -> None:
        self.obs_space = obs_space
        self.action_space = action_space

    def __repr__(self) -> str:  # lightweight to aid debugging
        return f"PolicyNetwork(obs_space={self.obs_space!r}, action_space={self.action_space!r})"


class ValueNetwork:  # pragma: no cover - placeholder
    """Stub for the value network."""

    def __init__(self, obs_space: Any) -> None:
        self.obs_space = obs_space

    def __repr__(self) -> str:
        return f"ValueNetwork(obs_space={self.obs_space!r})"
