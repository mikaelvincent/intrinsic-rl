"""Environment managers and wrappers (skeleton).

Vectorized environment creation, seeding, and domain randomization wrappers will be implemented in Sprint 0 — Step 4.
"""

from __future__ import annotations

from typing import Any, Sequence


class EnvManager:
    """Placeholder interface for the vectorized environment manager."""

    def __init__(self, env_id: str, num_envs: int = 1, **kwargs: Any) -> None:
        self.env_id = env_id
        self.num_envs = num_envs
        self.kwargs = kwargs

    def make(self) -> Any:  # pragma: no cover - to be implemented later
        """Create and return a vectorized environment.

        Returns:
            An object representing the vectorized environment.

        Raises:
            NotImplementedError: Until Sprint 0 Step 4.
        """
        raise NotImplementedError("EnvManager.make will be implemented in Step 4.")
