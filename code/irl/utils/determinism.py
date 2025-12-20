from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

__all__ = ["seed_everything"]

_TRUTHY_ENV: set[str] = {"1", "true", "yes", "y", "on"}
_DETERMINISTIC_WARN_ONLY_ENV = "IRL_DETERMINISTIC_WARN_ONLY"


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    return v.strip().lower() in _TRUTHY_ENV


def _device_is_cuda(device: Any | None) -> bool:
    if device is None:
        return False
    try:
        d = device if isinstance(device, torch.device) else torch.device(str(device))
    except Exception:
        return False
    return d.type == "cuda"


def seed_everything(
    seed: int,
    deterministic: bool = False,
    *,
    device: str | torch.device | None = None,
    warn_only: bool | None = None,
) -> None:
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(s)
        except Exception:
            pass

    if not deterministic:
        return

    on_cuda = _device_is_cuda(device) or (device is None and torch.cuda.is_available())
    use_warn_only = (
        bool(warn_only)
        if warn_only is not None
        else (on_cuda and _truthy_env(_DETERMINISTIC_WARN_ONLY_ENV))
    )

    try:
        try:
            torch.use_deterministic_algorithms(True, warn_only=bool(use_warn_only))
        except TypeError:
            torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    try:
        import torch.backends.cudnn as cudnn

        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
