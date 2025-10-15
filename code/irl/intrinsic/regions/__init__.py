"""Region data structures for intrinsic modules (R‑IAC / Proposed).

Currently provides a lightweight KD‑tree region store with split logic.
"""
from __future__ import annotations

from .kdtree import KDTreeRegionStore, RegionNode

__all__ = ["KDTreeRegionStore", "RegionNode"]
