from irl.utils.checkpoint import compute_cfg_hash


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
