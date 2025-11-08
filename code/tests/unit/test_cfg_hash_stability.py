from irl.utils.checkpoint import compute_cfg_hash


def test_config_hash_is_order_invariant_and_changes_on_diff():
    # Same logical config, different key order
    a = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 32, "steps_per_update": 128},
    }
    b = {
        "ppo": {"steps_per_update": 128, "minibatches": 32},
        "env": {"vec_envs": 8, "id": "MountainCar-v0"},
        "seed": 7,
        "method": "vanilla",
    }

    h_a = compute_cfg_hash(a)
    h_b = compute_cfg_hash(b)
    assert isinstance(h_a, str) and isinstance(h_b, str)
    assert h_a == h_b, "Hashes should match for identical content regardless of order"

    # Small semantic change -> hash must change
    c = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 64, "steps_per_update": 128},  # changed minibatches
    }
    h_c = compute_cfg_hash(c)
    assert h_c != h_a
