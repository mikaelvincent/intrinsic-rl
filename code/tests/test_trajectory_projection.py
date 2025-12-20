from __future__ import annotations

import numpy as np

from irl.visualization.trajectory_projection import trajectory_projection


def test_trajectory_projection_known_envs() -> None:
    obs4 = np.zeros((10, 4), dtype=np.float32)
    obs2 = np.zeros((10, 2), dtype=np.float32)

    assert trajectory_projection("MountainCar-v0", obs4) == (0, 1, "position", "velocity")
    assert trajectory_projection("CartPole-v1", obs4) == (0, 2, "cart_pos", "pole_angle")
    assert trajectory_projection("CartPole-v1", obs2) == (0, 1, "obs[0]", "obs[1]")
    assert trajectory_projection("Pendulum-v1", obs4) == (0, 1, "cos(theta)", "sin(theta)")
    assert trajectory_projection("Acrobot-v1", obs4) == (0, 1, "cos(theta1)", "sin(theta1)")
    assert trajectory_projection("LunarLander-v2", obs4) == (0, 1, "x", "y")

    assert trajectory_projection("BipedalWalker-v3", obs4) is None
    assert trajectory_projection("BipedalWalker-v3", obs4, include_bipedalwalker=True) == (
        0,
        1,
        "obs[0]",
        "obs[1]",
    )

    assert trajectory_projection("AnyEnv", np.zeros((10,), dtype=np.float32)) is None
    assert trajectory_projection("AnyEnv", np.zeros((10, 1), dtype=np.float32)) is None
