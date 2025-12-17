import numpy as np

from irl.intrinsic.regions.kdtree import KDTreeRegionStore


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
