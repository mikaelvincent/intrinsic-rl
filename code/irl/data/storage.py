"""Rollout storage abstractions used by PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class Transition:  # pragma: no cover - placeholder
    """Simple transition record used by the placeholder rollout buffer.

    Attributes
    ----------
    s : Any
        Observation at time :math:`t`.
    a : Any
        Action taken at time :math:`t`.
    r_ext : float
        Extrinsic reward collected after the action.
    s_next : Any
        Observation at time :math:`t + 1`.
    done : bool
        Whether the episode terminated after this transition.
    """

    s: Any
    a: Any
    r_ext: float
    s_next: Any
    done: bool = False


class RolloutBuffer:  # pragma: no cover - placeholder
    """Minimal in-memory rollout buffer.

    This class is a small example container used in tests and simple
    experiments. The main trainer uses more specialised storage, so this
    buffer intentionally only supports appending, clearing, and
    :func:`len`.
    """

    def __init__(self) -> None:
        self._data: List[Transition] = []

    def add(self, tr: Transition) -> None:
        """Append a transition to the buffer."""
        self._data.append(tr)

    def clear(self) -> None:
        """Remove all stored transitions."""
        self._data.clear()

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return len(self._data)
