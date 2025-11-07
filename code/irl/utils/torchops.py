"""Tiny torch helper utilities centralized for reuse.

Functions:
- as_tensor(x, device, dtype): like torch.as_tensor but moves to device and
  defaults dtype to float32 when x is not already a tensor.
- ensure_2d(t): ensure the last dimension is preserved and return a [B, D] view.
- one_hot(a, n): simple one-hot encoder for integer action indices.

These helpers factor out small, repeated snippets previously duplicated across
intrinsic modules (ICM, RIDE, RND, RIAC, Proposed).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor


def as_tensor(x: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Convert `x` to a Tensor on `device`, using `dtype` or fallback to float32.

    * If `x` is already a Tensor, it is moved to `device` and cast to `dtype` (if given).
    * Otherwise uses `torch.as_tensor(x, device=device, dtype=(dtype or float32))`.
    """
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype or torch.float32)


def ensure_2d(x: Tensor) -> Tensor:
    """Ensure a [B, D]-shaped view.

    - 1D -> [1, D]
    - 2D -> unchanged
    - ND (N>2) -> flatten all but last dim -> [*, D]
    """
    if x.dim() == 1:
        return x.view(1, -1)
    if x.dim() == 2:
        return x
    return x.view(-1, x.size(-1))


def one_hot(a: Tensor, n: int) -> Tensor:
    """Return a simple one-hot encoding for integer actions.

    Args:
        a: Tensor of integer indices (any shape, flattened to [B]).
        n: number of classes.
    Returns:
        Tensor [B, n] (float32) with 1.0 at class index, 0.0 elsewhere.
    """
    a = a.long().view(-1)
    out = torch.zeros((a.numel(), int(n)), device=a.device, dtype=torch.float32)
    out.scatter_(1, a.view(-1, 1), 1.0)
    return out
