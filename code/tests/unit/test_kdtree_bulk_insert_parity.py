import numpy as np

from irl.intrinsic.regions.kdtree import KDTreeRegionStore


def test_bulk_insert_matches_sequential_insert():
    rng = np.random.default_rng(0)
    dim = 3
    # Small capacity to trigger frequent splits
    store_seq = KDTreeRegionStore(dim=dim, capacity=4, depth_max=6)
    store_bulk = KDTreeRegionStore(dim=dim, capacity=4, depth_max=6)

    pts = rng.standard_normal((200, dim)).astype(np.float32)

    # Sequential inserts
    rids_seq = np.empty((pts.shape[0],), dtype=np.int64)
    for i in range(pts.shape[0]):
        rids_seq[i] = store_seq.insert(pts[i])

    # Bulk inserts
    rids_bulk = store_bulk.bulk_insert(pts)

    # The optimized bulk path must preserve the exact region-id sequence
    assert rids_seq.shape == rids_bulk.shape
    assert np.all(rids_seq == rids_bulk)

    # Trees should have comparable structure characteristics
    assert store_seq.num_regions() == store_bulk.num_regions()
    assert store_seq.max_depth() == store_bulk.max_depth()
