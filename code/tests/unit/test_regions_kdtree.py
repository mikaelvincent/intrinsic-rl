import numpy as np

from irl.intrinsic.regions.kdtree import KDTreeRegionStore


def test_kdtree_splits_at_capacity():
    store = KDTreeRegionStore(dim=2, capacity=4, depth_max=8)
    pts = np.array(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    for p in pts:
        store.insert(p)

    assert store.num_regions() == 2
    assert not store.root.is_leaf


def test_kdtree_selects_max_variance_dimension():
    store = KDTreeRegionStore(dim=2, capacity=4, depth_max=8)
    pts = np.array(
        [[1.0, 0.0], [1.0, 10.0], [1.0, 20.0], [1.0, 30.0], [1.0, 40.0]],
        dtype=np.float32,
    )
    for p in pts:
        store.insert(p)

    assert store.root.split_dim == 1


def test_kdtree_depth_limit_stops_splitting():
    store = KDTreeRegionStore(dim=2, capacity=2, depth_max=1)
    pts = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
        dtype=np.float32,
    )
    for p in pts:
        store.insert(p)

    assert store.num_regions() == 2


def test_bulk_insert_matches_sequential_insert():
    rng = np.random.default_rng(0)
    dim = 3
    pts = rng.standard_normal((100, dim)).astype(np.float32)

    store_seq = KDTreeRegionStore(dim=dim, capacity=4, depth_max=6)
    rids_seq = np.array([store_seq.insert(p) for p in pts], dtype=np.int64)

    store_bulk = KDTreeRegionStore(dim=dim, capacity=4, depth_max=6)
    rids_bulk = store_bulk.bulk_insert(pts)

    assert np.all(rids_seq == rids_bulk)
    assert store_seq.num_regions() == store_bulk.num_regions()


def test_bulk_insert_identical_points_stay_single_region():
    store = KDTreeRegionStore(dim=3, capacity=2, depth_max=4)
    pts = np.zeros((5, 3), dtype=np.float32)
    rids = store.bulk_insert(pts)

    assert np.all(rids == 0)
    assert store.num_regions() == 1
