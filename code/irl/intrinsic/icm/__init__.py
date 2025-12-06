"""ICM intrinsic module subpackage.

Public API: :class:`ICM` and :class:`ICMConfig`.
"""

from __future__ import annotations

import warnings

from .module import ICM, ICMConfig

__all__ = ["ICM", "ICMConfig"]


def __getattr__(name: str):
    """Deprecated aliases for older deep imports.

    - ``irl.intrinsic.icm.mlp`` → ``irl.intrinsic.icm.encoder.mlp``
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
