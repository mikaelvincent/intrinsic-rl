"""Proposed intrinsic module (refactor of monolith).

Files: core (Proposed), gating (_RegionStats/gate update), normalize (RMS helpers).
See devspec §5.4. Public API: `from irl.intrinsic.proposed import Proposed, _RegionStats`.
"""

from __future__ import annotations

from .core import Proposed
from .gating import _RegionStats

__all__ = ["Proposed", "_RegionStats"]
