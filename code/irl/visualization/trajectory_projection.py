from __future__ import annotations

import numpy as np


def trajectory_projection(
    env_id: str | None,
    obs: np.ndarray,
    *,
    include_bipedalwalker: bool = True,
) -> tuple[int, int, str, str] | None:
    if obs.ndim != 2 or obs.shape[1] < 2:
        return None

    D = int(obs.shape[1])
    e = (env_id or "").strip()

    if e.startswith("MountainCar"):
        return 0, 1, "position", "velocity"

    if bool(include_bipedalwalker) and e.startswith("BipedalWalker") and D >= 2:
        return 0, 1, "obs[0]", "obs[1]"

    if e.startswith(("Ant", "HalfCheetah", "Humanoid")) and D >= 2:
        return 0, 1, "obs[0]", "obs[1]"

    if D == 2:
        return 0, 1, "obs[0]", "obs[1]"

    return None
