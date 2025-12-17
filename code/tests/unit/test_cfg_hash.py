from irl.utils.checkpoint import compute_cfg_hash


def test_config_hash_is_order_invariant_and_sensitive_to_changes():
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
    assert compute_cfg_hash(a) == compute_cfg_hash(b)

    c = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 64, "steps_per_update": 128},
    }
    assert compute_cfg_hash(c) != compute_cfg_hash(a)


def test_config_hash_ignores_profile_cuda_sync():
    base = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 32, "steps_per_update": 128},
        "exp": {"deterministic": True},
    }
    h0 = compute_cfg_hash(base)

    with_flag = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 32, "steps_per_update": 128},
        "exp": {"deterministic": True, "profile_cuda_sync": True},
    }
    assert compute_cfg_hash(with_flag) == h0
