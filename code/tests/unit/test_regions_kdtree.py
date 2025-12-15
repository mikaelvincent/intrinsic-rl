import numpy as np

from irl.intrinsic.regions.kdtree import KDTreeRegionStore


def test_kdtree_split_triggers_at_capacity():
    store = KDTreeRegionStore(dim=2, capacity=4, depth_max=8)
    pts = np.array(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    for p in pts:
        store.insert(p)

    assert not store.root.is_leaf
    assert store.root.split_dim == 0

    left_cnt = store.root.left.count
    right_cnt = store.root.right.count
    assert left_cnt > 0 and right_cnt > 0
    assert left_cnt + right_cnt == pts.shape[0]


def test_kdtree_selects_max_variance_dimension():
    store = KDTreeRegionStore(dim=2, capacity=4, depth_max=8)
    pts = np.array(
        [[1.0, 0.0], [1.0, 10.0], [1.0, 20.0], [1.0, 30.0], [1.0, 40.0]],
        dtype=np.float32,
    )
    for p in pts:
        store.insert(p)
    assert store.root.split_dim == 1


def test_kdtree_depth_limit_prevents_further_splits():
    store = KDTreeRegionStore(dim=2, capacity=2, depth_max=1)
    pts = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
        dtype=np.float32,
    )
    for p in pts:
        store.insert(p)
    assert store.num_regions() == 2
    assert store.root.left.is_leaf and store.root.right.is_leaf


def test_kdtree_bbox_covers_points():
    store = KDTreeRegionStore(dim=2, capacity=3, depth_max=8)
    pts = np.array(
        [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
        dtype=np.float32,
    )
    for p in pts:
        store.insert(p)

    for leaf in store.iter_leaves():
        if leaf.count == 0:
            continue
        lo = leaf.bbox_lo
        hi = leaf.bbox_hi
        assert lo is not None and hi is not None
        pts_leaf = leaf.points()
        assert np.all(pts_leaf >= lo - 1e-6)
        assert np.all(pts_leaf <= hi + 1e-6)


def test_kdtree_locate_matches_insert_id():
    store = KDTreeRegionStore(dim=2, capacity=3, depth_max=8)
    pts = np.array(
        [[0.0, 0.0], [0.5, 0.1], [1.0, 0.2], [1.5, 0.3], [2.0, 0.4]],
        dtype=np.float32,
    )
    ids = [store.insert(p) for p in pts]
    for p, rid in zip(pts, ids):
        assert store.locate(p) == rid


def test_bulk_insert_handles_unsplittable_leaf():
    store = KDTreeRegionStore(dim=3, capacity=2, depth_max=4)
    pts = np.zeros((5, 3), dtype=np.float32)
    ids = store.bulk_insert(pts)

    assert np.all(ids == 0)
    assert store.num_regions() == 1
    assert store.root.is_leaf
    assert store.root.count == pts.shape[0]


def test_bulk_insert_handles_failed_split_signal():
    store = KDTreeRegionStore(dim=3, capacity=2, depth_max=4)

    def fake_split(leaf):
        return True

    store._split_leaf = fake_split

    pts = np.ones((3, 3), dtype=np.float32)
    ids = store.bulk_insert(pts)

    assert np.all(ids == 0)
    assert store.num_regions() == 1
