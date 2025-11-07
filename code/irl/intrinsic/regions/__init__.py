"""Region structures for intrinsic modules (KD‑tree store).

See devspec §5.6.
"""

from __future__ import annotations

from .kdtree import KDTreeRegionStore, RegionNode

__all__ = ["KDTreeRegionStore", "RegionNode"]
