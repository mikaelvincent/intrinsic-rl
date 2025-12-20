from __future__ import annotations

import numpy as np


def trajectory_projection(
    env_id: str | None,
    obs: np.ndarray,
    *,
    include_bipedalwalker: bool = False,
) -> tuple[int, int, str, str] | None:
    if obs.ndim != 2 or obs.shape[1] < 2:
        return None

    D = int(obs.shape[1])
    e = (env_id or "").strip()

    if e.startswith("MountainCar"):
        return 0, 1, "position", "velocity"
    if e.startswith("CartPole"):
        return (0, 2, "cart_pos", "pole_angle") if D >= 3 else (0, 1, "obs[0]", "obs[1]")
    if e.startswith("Pendulum"):
        return 0, 1, "cos(theta)", "sin(theta)"
    if e.startswith("Acrobot"):
        return 0, 1, "cos(theta1)", "sin(theta1)"
    if e.startswith("LunarLander"):
        return 0, 1, "x", "y"

    if include_bipedalwalker and e.startswith("BipedalWalker") and D >= 2:
        return 0, 1, "obs[0]", "obs[1]"

    if D == 2:
        return 0, 1, "obs[0]", "obs[1]"

    return None
