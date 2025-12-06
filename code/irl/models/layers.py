"""Common layers and builders for MLP backbones.

This module provides a small set of reusable building blocks:

* :class:`FlattenObs` — flattens all non-batch dimensions into a single
  feature axis so downstream networks can accept inputs shaped
  ``(D,)``, ``(B, D)`` or ``(T, B, D)``.
* :func:`mlp` — constructs a feedforward MLP with ReLU activations and
  an optional output projection layer.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

__all__ = ["FlattenObs", "mlp"]


class FlattenObs(nn.Module):
    """Flatten all non-batch dimensions into a single feature axis.

    This layer accepts tensors of rank >= 1 and returns a 2-D tensor
    of shape ``(B, F)``, where ``B`` is the effective batch size and
    ``F`` is the product of the remaining dimensions.

    Examples
    --------
    * ``(D,)``           → ``(1, D)``
    * ``(B, D)``         → unchanged
    * ``(T, B, D)``      → ``(T * B, D)``
    * ``(B, C, H, W)``   → ``(B, C * H * W)``
    """

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x.view(1, -1)
        return x.view(x.size(0), -1)


def mlp(
    in_dim: int,
    hidden: Sequence[int] = (256, 256),
    out_dim: int | None = None,
) -> nn.Sequential:
    """Build a feedforward MLP with ReLU activations.

    The returned network begins with :class:`FlattenObs`, so it can
    consume inputs with leading time or batch dimensions (for example,
    ``(T, B, D)`` or ``(B, D)``) and will internally reshape them to
    ``(N, in_dim)`` before applying the linear layers.

    Parameters
    ----------
    in_dim : int
        Size of the input feature vector fed into the first linear layer.
    hidden : Sequence[int], optional
        Hidden-layer sizes used to construct ``Linear → ReLU`` blocks.
    out_dim : int, optional
        Size of the output feature vector. When ``None``, the network
        ends with the last hidden layer and no explicit output layer.

    Returns
    -------
    torch.nn.Sequential
        Sequential module consisting of :class:`FlattenObs`, one or more
        ``Linear → ReLU`` blocks, and an optional final ``Linear`` layer.
    """
    layers: list[nn.Module] = [FlattenObs()]
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, int(h)), nn.ReLU(inplace=True)]
        last = int(h)
    if out_dim is not None:
        layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)
