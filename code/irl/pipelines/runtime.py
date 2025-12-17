from __future__ import annotations

from typing import Any

import numpy as np


def extract_env_runtime(cfg: Any) -> dict[str, object]:
    env_cfg = (cfg.get("env") or {}) if isinstance(cfg, dict) else {}
    return {
        "frame_skip": int(env_cfg.get("frame_skip", 1)),
        "discrete_actions": bool(env_cfg.get("discrete_actions", True)),
        "car_action_set": env_cfg.get("car_discrete_action_set", None),
    }


def build_obs_normalizer(payload: Any) -> tuple[np.ndarray, np.ndarray] | None:
    on = payload.get("obs_norm")
    if on is None:
        return None
    mean_arr = np.asarray(on.get("mean"), dtype=np.float64)
    var_arr = np.asarray(on.get("var"), dtype=np.float64)
    std_arr = np.sqrt(var_arr + 1e-8)
    return mean_arr, std_arr
