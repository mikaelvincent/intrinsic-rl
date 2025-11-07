"""ICM subpackage (refactor).

Public API: ICM, ICMConfig.
"""

from __future__ import annotations

import warnings

from .module import ICM, ICMConfig

__all__ = ["ICM", "ICMConfig"]


def __getattr__(name: str):
    """Legacy shims for deep/old imports with deprecation notice.

    - `irl.intrinsic.icm.mlp` → `irl.intrinsic.icm.encoder.mlp`
    """
    if name == "mlp":
        warnings.warn(
            "`irl.intrinsic.icm.mlp` is deprecated; use "
            "`irl.intrinsic.icm.encoder.mlp` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .encoder import mlp  # deferred import

        return mlp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
