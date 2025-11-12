"""Region data structures (KD-tree store) used by intrinsic modules.

Exports :class:`KDTreeRegionStore` and :class:`RegionNode` for φ-space
partitioning and statistics tracking.
"""

from __future__ import annotations

from .kdtree import KDTreeRegionStore, RegionNode

__all__ = ["KDTreeRegionStore", "RegionNode"]
