"""RND intrinsic module (skeleton)."""

from . import BaseIntrinsicModule, IntrinsicOutput, Transition


class RND(BaseIntrinsicModule):  # pragma: no cover - placeholder
    def compute(self, tr: Transition) -> IntrinsicOutput:
        raise NotImplementedError("RND will be implemented in a later sprint.")
