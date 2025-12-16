from irl.utils.checkpoint import compute_cfg_hash


def test_config_hash_is_order_invariant_and_changes_on_diff():
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

    h = compute_cfg_hash(a)
    assert compute_cfg_hash(b) == h

    c = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 64, "steps_per_update": 128},
    }
    assert compute_cfg_hash(c) != h
