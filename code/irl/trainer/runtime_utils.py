"""Small training-time utilities shared by the PPO loop.

This module hosts helpers that were previously defined inline in
``irl.trainer.loop`` and are now factored out to keep the main training
entry point smaller and easier to scan.

The helpers here are intentionally lightweight and logic-identical to
their original implementations.
"""

from __future__ import annotations

from typing import Optional

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
