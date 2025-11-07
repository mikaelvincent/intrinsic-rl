"""ICM heads: inverse (discrete/continuous) and forward dynamics."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .encoder import mlp


class ContinuousInverseHead(nn.Module):
    """Inverse head for continuous actions: predicts mean and (bounded) log_std."""

    def __init__(self, in_dim: int, act_dim: int, hidden: Iterable[int] = (256, 256)) -> None:
        super().__init__()
        self.backbone = mlp(in_dim, hidden, out_dim=2 * act_dim)
        self.log_std_min = -5.0
        self.log_std_max = 2.0
        self.act_dim = int(act_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.backbone(x)  # [B, 2*A]
        mu, log_std = torch.split(h, self.act_dim, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std


class ForwardHead(nn.Module):
    """Forward dynamics head predicting next embedding φ̂(s')."""

    def __init__(self, in_dim: int, out_dim: int, hidden: Iterable[int] = (256, 256)) -> None:
        super().__init__()
        self.net = mlp(in_dim, hidden, out_dim=out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
