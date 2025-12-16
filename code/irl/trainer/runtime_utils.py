from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.optim import Adam


def _is_image_space(space) -> bool:
    return hasattr(space, "shape") and len(space.shape) >= 2


def _move_optimizer_state_to_device(opt: Adam, device: torch.device) -> None:
    for state in opt.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _ensure_time_major_np(x: np.ndarray, T: int, B: int, name: str) -> np.ndarray:
    if x.ndim < 2:
        raise ValueError(f"{name}: expected at least 2 dims (T,B,...), got shape={x.shape}")
    t0, b0 = int(x.shape[0]), int(x.shape[1])
    if t0 == T and b0 == B:
        return x
    if t0 == B and b0 == T:
        return np.swapaxes(x, 0, 1)
    raise ValueError(
        f"{name}: inconsistent leading dims. Expected (T,B)=({T},{B}); got {tuple(x.shape[:2])}. "
        "Ensure time is the first axis and batch is second."
    )


def _apply_final_observation(next_obs: Any, done: Any, infos: Any) -> np.ndarray:
    obs = np.asarray(next_obs)
    done_mask = np.asarray(done, dtype=bool).reshape(-1)
    if not done_mask.any():
        return obs

    final = None
    if isinstance(infos, dict):
        final = infos.get("final_observation", None)
        if final is None:
            final = infos.get("final_observations", None)
    elif isinstance(infos, (list, tuple)):
        finals: list[Any] = []
        for inf in infos:
            if isinstance(inf, dict):
                finals.append(inf.get("final_observation", None))
            else:
                finals.append(None)
        final = finals if finals else None

    if final is None:
        return obs

    fixed = np.array(obs, copy=True)

    try:
        if isinstance(final, np.ndarray) and final.shape[:1] == (done_mask.shape[0],):
            for i in np.flatnonzero(done_mask):
                fo_i = final[i]
                if fo_i is not None:
                    fixed[i] = np.asarray(fo_i)
            return fixed

        if isinstance(final, (list, tuple)) and len(final) == done_mask.shape[0]:
            for i in np.flatnonzero(done_mask):
                fo_i = final[i]
                if fo_i is not None:
                    fixed[i] = np.asarray(fo_i)
            return fixed
    except Exception:
        pass

    if done_mask.shape[0] == 1:
        try:
            fixed[0] = np.asarray(final)
        except Exception:
            pass

    return fixed
