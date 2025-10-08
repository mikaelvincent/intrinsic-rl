"""ICM intrinsic module (skeleton)."""

from . import BaseIntrinsicModule, IntrinsicOutput, Transition


class ICM(BaseIntrinsicModule):  # pragma: no cover - placeholder
    def compute(self, tr: Transition) -> IntrinsicOutput:
        raise NotImplementedError("ICM will be implemented in a later sprint.")
