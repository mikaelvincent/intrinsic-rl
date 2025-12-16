from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from irl.utils.torchops import as_tensor

__all__ = ["ImagePreprocessConfig", "to_channels_first", "rgb_to_grayscale", "preprocess_image"]


@dataclass(frozen=True)
class ImagePreprocessConfig:
    grayscale: bool = True
    scale_uint8: bool = True
    normalize_mean: Optional[Union[float, Sequence[float]]] = None
    normalize_std: Optional[Union[float, Sequence[float]]] = None
    channels_first: bool = True


def to_channels_first(x: Union[np.ndarray, Tensor]) -> Tensor:
    t = torch.as_tensor(x)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.dim() == 3:
        if t.shape[0] in (1, 3, 4):
            return t
        return t.permute(2, 0, 1)
    if t.dim() == 4:
        if t.shape[1] in (1, 3, 4):
            return t
        return t.permute(0, 3, 1, 2)
    raise ValueError(f"Unsupported image tensor rank: {t.dim()}")


def rgb_to_grayscale(x: Tensor) -> Tensor:
    t = x
    was_batched = t.dim() == 4
    if t.dim() == 3:
        t = t.unsqueeze(0)

    if t.shape[1] == 4:
        t = t[:, :3]
    if t.shape[1] != 3:
        raise ValueError(f"Expected 3 (or 4) channels for RGB(A), got {t.shape[1]}")

    w = torch.tensor([0.299, 0.587, 0.114], dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
    g = (t * w).sum(dim=1, keepdim=True)
    return g if was_batched else g.squeeze(0)


def _maybe_scale_uint8_or_float255(t: Tensor, enable: bool) -> Tensor:
    if not enable:
        return t.to(dtype=torch.float32)

    if t.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        out = t.to(dtype=torch.float32) / 255.0
        return out.clamp_(0.0, 1.0)

    out = t.to(dtype=torch.float32)
    try:
        with torch.no_grad():
            max_val = torch.amax(out)
        if torch.isfinite(max_val) and float(max_val.item()) > 1.5:
            out = out / 255.0
        out = out.clamp_(0.0, 1.0)
    except Exception:
        out = out
    return out


def _apply_normalization(
    t: Tensor,
    mean: Optional[Union[float, Sequence[float]]],
    std: Optional[Union[float, Sequence[float]]],
) -> Tensor:
    if mean is None and std is None:
        return t

    m = torch.as_tensor(mean if mean is not None else 0.0, dtype=t.dtype, device=t.device)
    s = torch.as_tensor(std if std is not None else 1.0, dtype=t.dtype, device=t.device)

    if t.dim() == 3:
        if m.dim() == 0:
            m = m.view(1, 1, 1)
            s = s.view(1, 1, 1)
        else:
            m = m.view(-1, 1, 1)
            s = s.view(-1, 1, 1)
    elif t.dim() == 4:
        if m.dim() == 0:
            m = m.view(1, 1, 1, 1)
            s = s.view(1, 1, 1, 1)
        else:
            m = m.view(1, -1, 1, 1)
            s = s.view(1, -1, 1, 1)

    return (t - m) / (s + 1e-8)


def preprocess_image(
    x: Union[np.ndarray, Tensor],
    cfg: Optional[ImagePreprocessConfig] = None,
    *,
    device: Union[str, torch.device] = "cpu",
) -> Tensor:
    cfg = cfg or ImagePreprocessConfig()

    t = as_tensor(x, device=torch.device(device))
    t = to_channels_first(t)

    if cfg.grayscale:
        ch_idx = 1 if t.dim() == 4 else 0
        if t.shape[ch_idx] in (3, 4):
            t = rgb_to_grayscale(t)

    t = _maybe_scale_uint8_or_float255(t, enable=cfg.scale_uint8)
    t = _apply_normalization(t, cfg.normalize_mean, cfg.normalize_std)

    if cfg.channels_first:
        return t
    if t.dim() == 3:
        return t.permute(1, 2, 0)
    return t.permute(0, 2, 3, 1)
