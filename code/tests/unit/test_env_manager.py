import numpy as np


def test_env_manager_single_env_smoke():
    from irl.envs import EnvManager

    m = EnvManager(env_id="MountainCar-v0", num_envs=1, seed=123, frame_skip=1)
    env = m.make()
    try:
        obs, info = env.reset()
        # action space should be usable
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        assert isinstance(r, (int, float, np.floating))
    finally:
        env.close()


def test_env_manager_vector_env_smoke():
    from irl.envs import EnvManager

    m = EnvManager(env_id="MountainCar-v0", num_envs=2, seed=123, frame_skip=1)
    env = m.make()
    try:
        obs, infos = env.reset()
        # step with zeros
        actions = np.zeros((2,), dtype=int)
        obs, rewards, terms, truncs, infos = env.step(actions)
        assert rewards.shape == (2,)
    finally:
        env.close()
