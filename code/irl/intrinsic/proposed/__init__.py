"""Proposed intrinsic module package.

This package splits the original monolithic `proposed.py` into:
- core.py      : the `Proposed` class implementation
- gating.py    : region gate state container and gate update helpers
- normalize.py : lightweight wrappers around running RMS normalization

Public API (backward compatible):
    from irl.intrinsic.proposed import Proposed, _RegionStats
"""

from __future__ import annotations

from .core import Proposed
from .gating import _RegionStats

__all__ = ["Proposed", "_RegionStats"]
