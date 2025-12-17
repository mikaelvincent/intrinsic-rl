from irl.utils.checkpoint import compute_cfg_hash


def test_compute_cfg_hash_is_stable_and_selective():
    base = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 32, "steps_per_update": 128},
        "exp": {"deterministic": True},
    }
    reordered = {
        "ppo": {"steps_per_update": 128, "minibatches": 32},
        "env": {"vec_envs": 8, "id": "MountainCar-v0"},
        "seed": 7,
        "method": "vanilla",
        "exp": {"deterministic": True},
    }
    assert compute_cfg_hash(base) == compute_cfg_hash(reordered)

    changed = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 64, "steps_per_update": 128},
        "exp": {"deterministic": True},
    }
    assert compute_cfg_hash(changed) != compute_cfg_hash(base)

    with_profile_flag = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 32, "steps_per_update": 128},
        "exp": {"deterministic": True, "profile_cuda_sync": True},
    }
    assert compute_cfg_hash(with_profile_flag) == compute_cfg_hash(base)
