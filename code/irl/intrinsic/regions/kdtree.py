"""KD-tree region store for φ-space partitioning.

Implements a balanced-ish binary KD-tree over embeddings
:math:`φ ∈ ℝ^D` with capacity-based splits:

* Each leaf holds up to ``capacity`` samples. On overflow (and if
  ``depth < depth_max``) it splits on the maximum-variance dimension
  at the median value.
* Region IDs are assigned to leaves. On split, the left child inherits
  the parent's region ID and the right child receives a new ID, which
  improves stability of region identifiers over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Tuple, List

import numpy as np


Array = np.ndarray


@dataclass
class RegionNode:
    """KD-tree node.

    Leaf nodes store samples and simple bounding-box statistics.
    Internal nodes store split metadata and pointers to children.
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
        """Append a sample to this leaf and update its bounding box."""
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
        """Return a ``(N, D)`` array of stored samples (copy)."""
        if not self._points:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack(self._points, axis=0).astype(np.float32)

    # ------------------- split helpers -------------------

    def _split_candidates(self) -> List[int]:
        """Return dimensions sorted by descending sample variance."""
        pts = self.points()
        if pts.shape[0] == 0:
            return []
        var = pts.var(axis=0)
        order = np.argsort(var)[::-1]
        return [int(d) for d in order]

    def _median_and_masks(self, pts: Array, d: int) -> Tuple[float, Array, Array]:
        """Return median value and boolean masks for a split along dim ``d``."""
        vals = pts[:, d]
        med = float(np.median(vals))
        left_mask = vals <= med
        right_mask = vals > med
        return med, left_mask, right_mask

    # ------------------- query routing -------------------

    def route(self, p: Array) -> "RegionNode":
        """Descend from this node to the leaf appropriate for point ``p``."""
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
    """KD-tree region store with capacity-based splitting.

    Parameters
    ----------
    dim :
        Dimensionality of the embedding space.
    capacity :
        Maximum number of samples per leaf before a split is attempted.
    depth_max :
        Maximum depth of the tree. When reached, leaves stop splitting.
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
        """Insert a single point and return its region id.

        The returned region id reflects the leaf *after* any split
        triggered by the insertion.
        """
        leaf = self.root.route(p)
        leaf.add_point(p)
        if leaf.count >= self.capacity and leaf.depth < self.depth_max:
            self._split_leaf(leaf)
        final_leaf = self.root.route(p)
        assert final_leaf.region_id is not None
        return int(final_leaf.region_id)

    def bulk_insert(self, points: Array) -> Array:
        """Insert a batch of points and return their region ids.

        This method preserves the exact insertion semantics and
        region-id assignment order of the sequential :meth:`insert`
        implementation, while avoiding an extra root→leaf traversal
        per point. For each point it:

        1. routes to the current leaf,
        2. adds the point and splits if capacity is reached, and
        3. determines the final child locally using the split
           metadata (instead of re-routing from the root).
        """
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != self.dim:
            raise ValueError(f"points must have shape (N, {self.dim})")

        out = np.empty((pts.shape[0],), dtype=np.int64)

        for i in range(pts.shape[0]):
            x = pts[i]
            # 1) Route once to the current leaf
            leaf = self.root.route(x)
            # 2) Add the point and split if necessary
            leaf.add_point(x)
            if leaf.count >= self.capacity and leaf.depth < self.depth_max:
                # Cache split after mutation: _split_leaf turns `leaf` into an internal node
                self._split_leaf(leaf)
                # 3) Decide which child owns the newly added point using the split criterion
                assert leaf.split_dim is not None and leaf.split_val is not None
                assert leaf.left is not None and leaf.right is not None
                if float(x[leaf.split_dim]) <= float(leaf.split_val):
                    final_leaf = leaf.left
                else:
                    final_leaf = leaf.right
            else:
                final_leaf = leaf

            rid = final_leaf.region_id
            assert rid is not None
            out[i] = int(rid)

        return out

    def locate(self, p: Array) -> int:
        """Return the current region id (leaf) where point ``p`` would fall."""
        leaf = self.root.route(p)
        assert leaf.region_id is not None
        return int(leaf.region_id)

    # ------------------- tree stats -------------------

    def num_regions(self) -> int:
        """Return the number of leaf regions."""
        return len(self.leaf_by_id)

    def iter_leaves(self) -> Iterator[RegionNode]:
        """Iterate over all current leaf nodes."""
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
        """Split a leaf along its maximum-variance dimension at the median.

        Both children are guaranteed to be non-empty. When a suitable
        split cannot be found (e.g. degenerate data), the leaf is left
        unchanged.
        """
        assert leaf.is_leaf
        pts = leaf.points()
        if pts.shape[0] <= 1:
            return

        split_dim: Optional[int] = None
        split_val: Optional[float] = None
        left_mask = right_mask = None  # type: ignore[assignment]
        for d in leaf._split_candidates():
            med, lmask, rmask = leaf._median_and_masks(pts, d)
            if lmask.any() and rmask.any():
                split_dim, split_val = d, med
                left_mask, right_mask = lmask, rmask
                break

        if split_dim is None or split_val is None or left_mask is None or right_mask is None:
            return

        # Children (left inherits parent region_id; right gets a new one)
        left = RegionNode(
            depth=leaf.depth + 1,
            dim=leaf.dim,
            capacity=leaf.capacity,
            region_id=leaf.region_id,
        )
        right = RegionNode(
            depth=leaf.depth + 1,
            dim=leaf.dim,
            capacity=leaf.capacity,
            region_id=self._next_region_id,
        )
        self._next_region_id += 1

        for x in pts[left_mask]:
            left.add_point(x)
        for x in pts[right_mask]:
            right.add_point(x)

        # Update topology and registry
        leaf.is_leaf = False
        leaf.left = left
        leaf.right = right
        leaf.split_dim = split_dim
        leaf.split_val = float(split_val)
        leaf._points.clear()
        leaf.count = 0
        leaf.bbox_lo = None
        leaf.bbox_hi = None

        assert left.region_id is not None and right.region_id is not None
        self.leaf_by_id[left.region_id] = left
        self.leaf_by_id[right.region_id] = right

    # ------------------- debug/diagnostics -------------------

    def as_dict(self) -> dict:
        """Return a JSON-serialisable snapshot of the tree structure."""

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
