from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    s = int(seed)

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(s)
        except Exception:
            pass

    if deterministic:
        try:
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
