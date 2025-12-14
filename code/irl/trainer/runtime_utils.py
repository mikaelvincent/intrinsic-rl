"""Small training-time utilities shared by the PPO loop.

This module hosts helpers that were previously defined inline in
``irl.trainer.loop`` and are now factored out to keep the main training
entry point smaller and easier to scan.

The helpers here are intentionally lightweight and logic-identical to
their original implementations.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from torch.optim import Adam


def _is_image_space(space) -> bool:
    """Return True if an observation space should be treated as image-like.

    Heuristic
    ---------
    A space is considered image-like when it exposes a ``shape`` attribute
    and that shape has rank ≥ 2.
    """
    return hasattr(space, "shape") and len(space.shape) >= 2


def _move_optimizer_state_to_device(opt: Adam, device: torch.device) -> None:
    """Move all optimizer state tensors in ``opt`` onto ``device``.

    This is used when restoring optimizer state from a checkpoint that
    was saved on a different device (for example CPU → CUDA).
    """
    for state in opt.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _ensure_time_major_np(x: np.ndarray, T: int, B: int, name: str) -> np.ndarray:
    """Return array with leading dims (T, B, ...) from (T, B, ...) or (B, T, ...).

    If the array is in batch-major layout (B, T, ...) it is automatically
    transposed to time-major (T, B, ...). Any other mismatch raises a
    ValueError with a clear message.

    Parameters
    ----------
    x:
        Input array that is expected to carry time and batch dimensions in
        its first two axes.
    T, B:
        Expected time and batch sizes.
    name:
        Human-readable name used in error messages (for example ``"obs_seq"``).

    Returns
    -------
    numpy.ndarray
        Array with leading dimensions ``(T, B, ...)``.
    """
    if x.ndim < 2:
        raise ValueError(f"{name}: expected at least 2 dims (T,B,...), got shape={x.shape}")
    t0, b0 = int(x.shape[0]), int(x.shape[1])
    if t0 == T and b0 == B:
        return x
    if t0 == B and b0 == T:
        # Auto-fix common mistake: batch-major provided instead of time-major.
        return np.swapaxes(x, 0, 1)
    raise ValueError(
        f"{name}: inconsistent leading dims. Expected (T,B)=({T},{B}); got {tuple(x.shape[:2])}. "
        "Ensure time is the first axis and batch is second."
    )


def _apply_final_observation(next_obs: Any, done: Any, infos: Any) -> np.ndarray:
    """Replace auto-reset observations with terminal observations when available.

    Gymnasium VectorEnv implementations can auto-reset individual environments
    when they terminate or truncate. In that mode, the observation returned by
    ``env.step()`` for a done environment may be the *reset* observation, while
    the true terminal observation is provided separately via:

        infos["final_observation"]  (and sometimes infos["final_info"])

    This helper returns a NumPy array where entries corresponding to ``done=True``
    are substituted with the terminal observations when available.

    Parameters
    ----------
    next_obs:
        Observation returned by ``env.step()`` (possibly already reset for done envs).
    done:
        Boolean done mask (terminations OR truncations).
    infos:
        Info payload returned by ``env.step()``. Commonly a dict for VectorEnv;
        sometimes a list of per-env dicts.

    Returns
    -------
    numpy.ndarray
        Next observations suitable for rollout storage / bootstrapping.
        Returns the original array view when no substitution is possible.
    """
    obs = np.asarray(next_obs)
    done_mask = np.asarray(done, dtype=bool).reshape(-1)

    # Fast path: nothing is done => no substitution needed.
    if not done_mask.any():
        return obs

    # Try to locate final observations in the info payload.
    final = None
    if isinstance(infos, dict):
        final = infos.get("final_observation", None)
        if final is None:
            # Defensive: tolerate pluralized variants if a backend uses them.
            final = infos.get("final_observations", None)
    elif isinstance(infos, (list, tuple)):
        # Some wrappers return a list of per-env info dicts.
        finals: list[Any] = []
        for inf in infos:
            if isinstance(inf, dict):
                finals.append(inf.get("final_observation", None))
            else:
                finals.append(None)
        final = finals if finals else None

    if final is None:
        return obs

    # Make a writable copy only when we actually attempt substitution.
    fixed = np.array(obs, copy=True)

    # Standard vector-env case: final is list/array with leading env axis.
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
        # Fall back to scalar assignment below.
        pass

    # Single-env case: treat final as a single observation payload.
    if done_mask.shape[0] == 1:
        try:
            fixed[0] = np.asarray(final)
        except Exception:
            pass

    return fixed
