import numpy as np


def test_env_manager_single_env_smoke():
    from irl.envs import EnvManager

    env = EnvManager(env_id="MountainCar-v0", num_envs=1, seed=123, frame_skip=1).make()
    try:
        env.reset()
        a = env.action_space.sample()
        _, r, _, _, _ = env.step(a)
        assert isinstance(r, (int, float, np.floating))
    finally:
        env.close()


def test_env_manager_vector_env_smoke():
    from irl.envs import EnvManager

    env = EnvManager(env_id="MountainCar-v0", num_envs=2, seed=123, frame_skip=1).make()
    try:
        env.reset()
        actions = np.zeros((2,), dtype=int)
        _, rewards, _, _, _ = env.step(actions)
        assert rewards.shape == (2,)
    finally:
        env.close()
