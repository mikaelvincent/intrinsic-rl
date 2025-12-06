"""R-IAC intrinsic module subpackage.

Public API: :class:`RIAC` and :func:`simulate_lp_emas`.
"""

from __future__ import annotations

import warnings

from .module import RIAC, simulate_lp_emas

__all__ = ["RIAC", "simulate_lp_emas"]


def __getattr__(name: str):
    """Legacy shim to keep old deep import working with a warning.

    - `export_diagnostics` (was co-located) → `riac.diagnostics.export_diagnostics`
    """
    if name == "export_diagnostics":
        warnings.warn(
            "`irl.intrinsic.riac.export_diagnostics` moved to "
            "`irl.intrinsic.riac.diagnostics.export_diagnostics`.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .diagnostics import export_diagnostics  # deferred import

        return export_diagnostics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
