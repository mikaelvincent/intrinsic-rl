from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Transition(Protocol):
    s: Any
    a: Any
    r_ext: float
    s_next: Any


@dataclass
class IntrinsicOutput:
    r_int: float


class BaseIntrinsicModule:
    def compute(self, tr: Transition) -> IntrinsicOutput:
        raise NotImplementedError


# Late imports avoid cycles.
from .factory import (
    compute_intrinsic_batch,
    create_intrinsic_module,
    is_intrinsic_method,
    update_module,
)
from .normalization import RunningRMS
from .glpe import GLPE
from .riac import RIAC
from .ride import RIDE

__all__ = [
    "Transition",
    "IntrinsicOutput",
    "BaseIntrinsicModule",
    "is_intrinsic_method",
    "create_intrinsic_module",
    "compute_intrinsic_batch",
    "update_module",
    "RunningRMS",
    "RIDE",
    "RIAC",
    "GLPE",
]
