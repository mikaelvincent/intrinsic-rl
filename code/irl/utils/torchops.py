"""Small torch helpers shared across modules.

Exposes :func:`as_tensor`, :func:`ensure_2d`, and :func:`one_hot` utilities used
in trainers and intrinsic modules.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor


def as_tensor(x: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Convert an input to a :class:`torch.Tensor` on a target device.

    Parameters
    ----------
    x : Any
        Array-like object or existing tensor to convert.
    device : torch.device
        Device on which the returned tensor should live.
    dtype : torch.dtype, optional
        Optional dtype override. When omitted, existing tensors keep their
        original dtype and non-tensor inputs default to ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Tensor view of ``x`` on ``device``.
    """
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype or torch.float32)


def ensure_2d(x: Tensor) -> Tensor:
    """Return a 2-D ``[B, D]`` view of a tensor.

    1-D inputs are expanded to ``[1, D]``, 2-D inputs are returned as-is, and
    higher-rank inputs are flattened along all leading dimensions except the
    last feature dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary rank ``>= 1``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[B, D]`` where ``D`` is the size of the last
        dimension of ``x``.
    """
    if x.dim() == 1:
        return x.view(1, -1)
    if x.dim() == 2:
        return x
    return x.view(-1, x.size(-1))


def one_hot(a: Tensor, n: int) -> Tensor:
    """One-hot encode integer indices into a dense matrix.

    Parameters
    ----------
    a : torch.Tensor
        Tensor of integer class indices with arbitrary shape. The input is
        reshaped to a 1-D vector before encoding.
    n : int
        Number of classes; determines the size of the one-hot dimension.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``[B, n]`` containing one-hot encodings of
        ``a``, where ``B`` is the number of indices.
    """
    a = a.long().view(-1)
    out = torch.zeros((a.numel(), int(n)), device=a.device, dtype=torch.float32)
    out.scatter_(1, a.view(-1, 1), 1.0)
    return out
