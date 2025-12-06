"""Lightweight MLP builder used by ICM encoder/heads."""

from __future__ import annotations

from typing import Iterable, Optional

from torch import nn


def mlp(in_dim: int, hidden: Iterable[int], out_dim: Optional[int] = None) -> nn.Sequential:
    """Construct a feedforward MLP used by the ICM encoder and heads.

    Parameters
    ----------
    in_dim : int
        Size of the input feature vector.
    hidden : Iterable[int]
        Sequence of hidden-layer sizes.
    out_dim : int, optional
        Size of the output feature vector. When ``None``, the network
        ends with the last hidden layer and no explicit output layer.

    Returns
    -------
    torch.nn.Sequential
        Sequential module consisting of Linear/ReLU blocks and an
        optional final Linear layer.
    """
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, int(h)), nn.ReLU(inplace=True)]
        last = int(h)
    if out_dim is not None:
        layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)
