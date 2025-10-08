"""Proposed unified intrinsic module (skeleton)."""

from . import BaseIntrinsicModule, IntrinsicOutput, Transition


class Proposed(BaseIntrinsicModule):  # pragma: no cover - placeholder
    def compute(self, tr: Transition) -> IntrinsicOutput:
        raise NotImplementedError(
            "Proposed module will be implemented in a later sprint."
        )
