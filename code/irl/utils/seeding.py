from __future__ import annotations

import torch

from irl.utils.determinism import seed_everything as _seed_everything

__all__ = ["seed_torch_only", "seed_all"]


def seed_torch_only(seed: int) -> None:
    s = int(seed)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(s)
        except Exception:
            pass


def seed_all(seed: int) -> None:
    _seed_everything(int(seed), deterministic=False)
