from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def to_tensor(x: Any, device: torch.device, dtype: torch.dtype | None = None) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype or x.dtype)
    if dtype is None:
        return torch.as_tensor(x, device=device)
    return torch.as_tensor(x, device=device, dtype=dtype)
