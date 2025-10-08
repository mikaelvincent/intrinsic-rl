"""Rollout storage abstractions (skeleton)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class Transition:  # pragma: no cover - placeholder
    s: Any
    a: Any
    r_ext: float
    s_next: Any
    done: bool = False


class RolloutBuffer:  # pragma: no cover - placeholder
    """Minimal placeholder interface for a rollout buffer."""

    def __init__(self) -> None:
        self._data: List[Transition] = []

    def add(self, tr: Transition) -> None:
        self._data.append(tr)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
