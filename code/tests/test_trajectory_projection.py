from __future__ import annotations

import numpy as np

from irl.visualization.trajectory_projection import trajectory_projection


def test_trajectory_projection_known_envs() -> None:
    obs4 = np.zeros((10, 4), dtype=np.float32)
    obs2 = np.zeros((10, 2), dtype=np.float32)

    assert trajectory_projection("MountainCar-v0", obs4) == (0, 1, "position", "velocity")

    assert trajectory_projection("BipedalWalker-v3", obs4) == (0, 1, "obs[0]", "obs[1]")
    assert trajectory_projection("BipedalWalker-v3", obs4, include_bipedalwalker=False) is None

    assert trajectory_projection("Ant-v5", obs4) == (0, 1, "obs[0]", "obs[1]")
    assert trajectory_projection("HalfCheetah-v5", obs4) == (0, 1, "obs[0]", "obs[1]")
    assert trajectory_projection("Humanoid-v5", obs4) == (0, 1, "obs[0]", "obs[1]")

    assert trajectory_projection("AnyEnv", obs2) == (0, 1, "obs[0]", "obs[1]")

    assert trajectory_projection("AnyEnv", np.zeros((10,), dtype=np.float32)) is None
    assert trajectory_projection("AnyEnv", np.zeros((10, 1), dtype=np.float32)) is None
