"""Lightweight MLP builder used by ICM encoder/heads."""

from __future__ import annotations

from typing import Iterable, Optional

from torch import nn


def mlp(in_dim: int, hidden: Iterable[int], out_dim: Optional[int] = None) -> nn.Sequential:
    """Simple MLP: Linear/ReLU stacks with optional final Linear."""
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, int(h)), nn.ReLU(inplace=True)]
        last = int(h)
    if out_dim is not None:
        layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)
