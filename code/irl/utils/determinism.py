"""Determinism helpers.

Provides a single place to seed Python's ``random``, NumPy, and PyTorch, with an
optional switch for stricter deterministic behavior in PyTorch.

Usage
-----
>>> from irl.utils.determinism import seed_everything
>>> seed_everything(123, deterministic=True)
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


__all__ = ["seed_everything"]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Optionally requests deterministic behaviour from PyTorch where supported.

    Parameters
    ----------
    seed:
        Seed value applied across libraries.
    deterministic:
        If True, request deterministic PyTorch behavior where supported.
        This may disable some fast kernels or raise if a non-deterministic
        op is used.
    """
    s = int(seed)

    # Python & NumPy
    random.seed(s)
    np.random.seed(s)

    # PyTorch (CPU/GPU)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(s)
        except Exception:
            # Be robust on CPU-only builds or unusual runtimes.
            pass

    if deterministic:
        # Best-effort deterministic settings for PyTorch.
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older versions may not support this API; ignore.
            pass
        try:
            import torch.backends.cudnn as cudnn  # noqa: WPS433 (runtime import OK)

            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass

        # CUDA matmul determinism (must be set before first CUDA context usage).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
