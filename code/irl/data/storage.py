from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Transition:
    s: Any
    a: Any
    r_ext: float
    s_next: Any
    done: bool = False


class RolloutBuffer:
    def __init__(self) -> None:
        self._data: list[Transition] = []

    def add(self, tr: Transition) -> None:
        self._data.append(tr)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
