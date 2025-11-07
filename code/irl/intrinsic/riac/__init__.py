"""RIAC package (split from monolithic module).

Public API:
    from irl.intrinsic.riac import RIAC, simulate_lp_emas
"""

from __future__ import annotations

from .module import RIAC, simulate_lp_emas

__all__ = ["RIAC", "simulate_lp_emas"]
