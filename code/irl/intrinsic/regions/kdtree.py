"""KD‑tree region store with split logic.

This module implements a *balanced-ish* binary KD‑tree over an embedding space φ∈R^D.
It is designed for R‑IAC / Proposed intrinsic methods to maintain **regions** of the
latent space with a simple capacity‑based split policy:

* Each **leaf** holds up to `capacity` samples; when exceeded and `depth < depth_max`,
  it **splits** on the dimension with the largest sample variance, at the **median**
  of that dimension. Points with value `<= median` go left; `> median` go right.
* If no split produces non‑empty children (e.g., all points identical), the node
  remains a leaf (no further splitting), and it may continue to accumulate samples.

Design notes
------------
* **Region IDs** are assigned **only to leaves**. When a leaf splits, the **left child
  inherits** the parent's region_id; the right child receives a **new** region_id.
  This ensures existing region_id references remain valid.
* Leaf nodes keep a small list of their raw samples (NumPy arrays) to support split
  statistics. With default `capacity=200`, memory remains modest and well‑bounded.
* Bounding boxes (lo/hi) are maintained per leaf for diagnostics/visualization.

Typical usage
-------------
>>> store = KDTreeRegionStore(dim=128, capacity=200, depth_max=12)
>>> rid = store.insert(phi)                 # phi: (128,) numpy array
>>> leaf = store.leaf_by_id[rid]
>>> store.num_regions()
42
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class RegionNode:
    """A node in the KD‑tree.

    Leaves hold samples; internal nodes hold split metadata.
    """

    depth: int
    dim: int
    capacity: int
    region_id: Optional[int] = None  # only meaningful for leaves
    # Tree topology
    is_leaf: bool = True
    left: Optional["RegionNode"] = None
    right: Optional["RegionNode"] = None
    split_dim: Optional[int] = None
    split_val: Optional[float] = None
    # Leaf storage/stats
    count: int = 0
    bbox_lo: Optional[Array] = None
    bbox_hi: Optional[Array] = None
    _points: List[Array] = field(default_factory=list)

    # ------------------- leaf operations -------------------

    def add_point(self, p: Array) -> None:
        """Append a sample to this leaf and update the bounding box."""
        assert self.is_leaf, "add_point only valid on leaves"
        x = np.asarray(p, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"Point has dim {x.shape[0]}, expected {self.dim}")
        self._points.append(x)
        self.count += 1
        if self.bbox_lo is None:
            self.bbox_lo = x.copy()
            self.bbox_hi = x.copy()
        else:
            self.bbox_lo = np.minimum(self.bbox_lo, x)
            self.bbox_hi = np.maximum(self.bbox_hi, x)

    def points(self) -> Array:
        """Return a (N, D) view of stored samples (copy)."""
        if not self._points:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack(self._points, axis=0).astype(np.float32)

    # ------------------- split helpers -------------------

    def _split_candidates(self) -> List[int]:
        """Return dimensions sorted by descending sample variance."""
        pts = self.points()
        if pts.shape[0] == 0:
            return []
        var = pts.var(axis=0)  # population variance
        order = np.argsort(var)[::-1]  # largest variance first
        return [int(d) for d in order]

    def _median_and_masks(self, pts: Array, d: int) -> Tuple[float, Array, Array]:
        vals = pts[:, d]
        med = float(np.median(vals))
        left_mask = vals <= med
        right_mask = vals > med
        return med, left_mask, right_mask

    # ------------------- query routing -------------------

    def route(self, p: Array) -> "RegionNode":
        """Descend to the leaf appropriate for point `p` (from this node)."""
        node: RegionNode = self
        x = np.asarray(p, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"Point has dim {x.shape[0]}, expected {self.dim}")
        while not node.is_leaf:
            assert node.split_dim is not None and node.split_val is not None
            if x[node.split_dim] <= node.split_val:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node


class KDTreeRegionStore:
    """KD‑tree region store with capacity‑based splitting.

    Parameters
    ----------
    dim : int
        Dimensionality of the embedding space φ.
    capacity : int
        Maximum samples per leaf before splitting is attempted.
    depth_max : int
        Maximum tree depth (root depth = 0). Leaves at depth >= depth_max will not split.
    """

    def __init__(self, dim: int, capacity: int = 200, depth_max: int = 12) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if depth_max < 0:
            raise ValueError("depth_max must be >= 0")

        self.dim = int(dim)
        self.capacity = int(capacity)
        self.depth_max = int(depth_max)

        # Root is a leaf to start with and gets region_id=0
        self.root = RegionNode(depth=0, dim=self.dim, capacity=self.capacity, region_id=0)
        self.leaf_by_id: Dict[int, RegionNode] = {0: self.root}
        self._next_region_id: int = 1

    # ------------------- basic API -------------------

    def insert(self, p: Array) -> int:
        """Insert a single point; return the (leaf) region_id it belongs to **after** any split."""
        leaf = self.root.route(p)
        leaf.add_point(p)
        if leaf.count > self.capacity and leaf.depth < self.depth_max:
            self._split_leaf(leaf)
        # Route again after potential split to return final region id
        final_leaf = self.root.route(p)
        assert final_leaf.region_id is not None
        return int(final_leaf.region_id)

    def bulk_insert(self, points: Array) -> Array:
        """Insert a batch of points; returns an array of region_ids for each point."""
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != self.dim:
            raise ValueError(f"points must have shape (N, {self.dim})")
        out = np.empty((pts.shape[0],), dtype=np.int64)
        for i in range(pts.shape[0]):
            out[i] = self.insert(pts[i])
        return out

    def locate(self, p: Array) -> int:
        """Return the current region_id (leaf) where point `p` would fall."""
        leaf = self.root.route(p)
        assert leaf.region_id is not None
        return int(leaf.region_id)

    # ------------------- tree stats -------------------

    def num_regions(self) -> int:
        """Number of leaf regions."""
        return len(self.leaf_by_id)

    def iter_leaves(self) -> Iterator[RegionNode]:
        """Iterate over current leaves."""
        yield from self.leaf_by_id.values()

    def max_depth(self) -> int:
        """Return the maximum depth among all leaves."""

        def _walk(n: RegionNode) -> int:
            if n.is_leaf:
                return n.depth
            return max(_walk(n.left), _walk(n.right))  # type: ignore[arg-type]

        return _walk(self.root)

    # ------------------- internal: splitting -------------------

    def _split_leaf(self, leaf: RegionNode) -> None:
        """Split a leaf node into two children based on median along a high‑variance dim."""
        assert leaf.is_leaf
        pts = leaf.points()
        # Defensive: if not enough points or all identical, skip
        if pts.shape[0] <= 1:
            return

        # Try dimensions by decreasing variance and choose the first that yields non‑empty children
        split_dim: Optional[int] = None
        split_val: Optional[float] = None
        left_mask = right_mask = None  # type: ignore[assignment]
        for d in leaf._split_candidates():
            med, lmask, rmask = leaf._median_and_masks(pts, d)
            if lmask.any() and rmask.any():
                split_dim, split_val = d, med
                left_mask, right_mask = lmask, rmask
                break

        # If no valid split exists, keep as a leaf (likely all points identical)
        if split_dim is None or split_val is None or left_mask is None or right_mask is None:
            return

        # Create children
        left = RegionNode(
            depth=leaf.depth + 1,
            dim=leaf.dim,
            capacity=leaf.capacity,
            region_id=leaf.region_id,  # inherit parent's id
        )
        right = RegionNode(
            depth=leaf.depth + 1,
            dim=leaf.dim,
            capacity=leaf.capacity,
            region_id=self._next_region_id,  # allocate new id
        )
        self._next_region_id += 1

        # Distribute points
        for x in pts[left_mask]:
            left.add_point(x)
        for x in pts[right_mask]:
            right.add_point(x)

        # Update tree topology
        leaf.is_leaf = False
        leaf.left = left
        leaf.right = right
        leaf.split_dim = split_dim
        leaf.split_val = float(split_val)
        # Clear leaf payloads/stats (internal nodes don't store samples)
        leaf._points.clear()
        leaf.count = 0
        leaf.bbox_lo = None
        leaf.bbox_hi = None

        # Update leaves registry: parent id now maps to left; add right
        assert left.region_id is not None and right.region_id is not None
        self.leaf_by_id[left.region_id] = left
        self.leaf_by_id[right.region_id] = right

    # ------------------- debug/diagnostics -------------------

    def as_dict(self) -> dict:
        """Return a JSON‑serializable snapshot of the tree (structure only)."""

        def _node(n: RegionNode) -> dict:
            d = {
                "depth": n.depth,
                "is_leaf": n.is_leaf,
                "region_id": n.region_id,
                "count": n.count if n.is_leaf else 0,
            }
            if not n.is_leaf:
                d["split_dim"] = n.split_dim
                d["split_val"] = n.split_val
                d["left"] = _node(n.left) if n.left is not None else None
                d["right"] = _node(n.right) if n.right is not None else None
            else:
                d["bbox_lo"] = None if n.bbox_lo is None else n.bbox_lo.tolist()
                d["bbox_hi"] = None if n.bbox_hi is None else n.bbox_hi.tolist()
            return d

        return {
            "dim": self.dim,
            "capacity": self.capacity,
            "depth_max": self.depth_max,
            "root": _node(self.root),
        }
