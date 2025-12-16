from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor


def as_tensor(x: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype or torch.float32)


def ensure_2d(x: Tensor) -> Tensor:
    if x.dim() == 1:
        return x.view(1, -1)
    if x.dim() == 2:
        return x
    return x.view(-1, x.size(-1))


def one_hot(a: Tensor, n: int) -> Tensor:
    a = a.long().view(-1)
    out = torch.zeros((a.numel(), int(n)), device=a.device, dtype=torch.float32)
    out.scatter_(1, a.view(-1, 1), 1.0)
    return out
