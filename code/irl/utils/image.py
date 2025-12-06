"""Image preprocessing utilities for RL pipelines.

Provides a small, dependency-free set of helpers to:

* Convert NHWC/NCHW to a canonical layout.
* Optionally convert RGB/RGBA → grayscale.
* Scale integer images from [0, 255] → [0, 1].
* If floats appear to be in [0, 255], scale to [0, 1] defensively.
* Apply simple mean/std normalization.
* Return a ``torch.Tensor`` suitable for CNNs ([N, C, H, W], float32).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from irl.utils.torchops import as_tensor


__all__ = [
    "ImagePreprocessConfig",
    "to_channels_first",
    "rgb_to_grayscale",
    "preprocess_image",
]


@dataclass(frozen=True)
class ImagePreprocessConfig:
    """Configuration for image preprocessing."""

    grayscale: bool = True
    scale_uint8: bool = True
    normalize_mean: Optional[Union[float, Sequence[float]]] = None
    normalize_std: Optional[Union[float, Sequence[float]]] = None
    channels_first: bool = True  # output layout
    # If provided, images will be resized/cropped upstream; this module does not resample.


def to_channels_first(x: Union[np.ndarray, Tensor]) -> Tensor:
    """Return a tensor in CHW (or NCHW) order from HWC/NHWC/CHW/NCHW inputs."""
    t = torch.as_tensor(x)
    if t.dim() == 2:
        # H, W -> 1, H, W
        t = t.unsqueeze(0)
    if t.dim() == 3:
        # CHW or HWC
        if t.shape[0] in (1, 3, 4):
            return t  # CHW
        return t.permute(2, 0, 1)  # HWC -> CHW
    if t.dim() == 4:
        # NCHW or NHWC
        if t.shape[1] in (1, 3, 4):
            return t  # NCHW
        return t.permute(0, 3, 1, 2)  # NHWC -> NCHW
    raise ValueError(f"Unsupported image tensor rank: {t.dim()}")


def rgb_to_grayscale(x: Tensor) -> Tensor:
    """Convert RGB(A) to grayscale using ITU-R BT.601 luma weights.

    Accepts CHW or NCHW. If 4 channels are present, the alpha channel is
    ignored. Returns a tensor with a single channel (C=1).
    """
    t = x
    was_batched = t.dim() == 4
    if t.dim() == 3:
        t = t.unsqueeze(0)  # -> NCHW

    if t.shape[1] == 4:
        t = t[:, :3]  # drop alpha
    if t.shape[1] != 3:
        raise ValueError(f"Expected 3 (or 4) channels for RGB(A), got {t.shape[1]}")

    # Weights: R=0.299, G=0.587, B=0.114
    w = torch.tensor([0.299, 0.587, 0.114], dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
    g = (t * w).sum(dim=1, keepdim=True)  # [N, 1, H, W]

    return g if was_batched else g.squeeze(0)


def _maybe_scale_uint8_or_float255(t: Tensor, enable: bool) -> Tensor:
    """Return a float32 image tensor and optionally normalise to [0, 1].

    When ``enable`` is True:

    * Integer types are scaled by 1/255.
    * Float types that appear to be in 0..255 range (max > ~1.5) are
      scaled by 1/255 as a defensive normalisation step.

    After optional scaling, values are clamped to [0, 1] for numeric safety.
    When ``enable`` is False the input is cast to float32 without scaling.
    """
    if not enable:
        return t.to(dtype=torch.float32)

    # Integer → [0,1]
    if t.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        out = t.to(dtype=torch.float32) / 255.0
        # Clamp for numeric safety.
        return out.clamp_(0.0, 1.0)

    # Float path
    out = t.to(dtype=torch.float32)
    try:
        # Use amax/amin; safe on any shape.
        with torch.no_grad():
            max_val = torch.amax(out)
            min_val = torch.amin(out)
        # Heuristic: if clearly not already in [0,1], assume 0..255 and scale.
        if torch.isfinite(max_val) and float(max_val.item()) > 1.5:
            out = out / 255.0
        # Clamp to [0,1] either way to keep downstream stable.
        out = out.clamp_(0.0, 1.0)
    except Exception:
        # Ultra-defensive: if reduction somehow fails, just return float32 cast.
        out = out
    return out


def _apply_normalization(
    t: Tensor,
    mean: Optional[Union[float, Sequence[float]]],
    std: Optional[Union[float, Sequence[float]]],
) -> Tensor:
    """Apply per-channel mean/std normalisation if requested."""
    if mean is None and std is None:
        return t
    # Broadcast-friendly: accept scalars or sequences.
    m = torch.as_tensor(mean if mean is not None else 0.0, dtype=t.dtype, device=t.device)
    s = torch.as_tensor(std if std is not None else 1.0, dtype=t.dtype, device=t.device)
    # Shape to [C, 1, 1] or [1, C, 1, 1] depending on rank.
    if t.dim() == 3:
        # CHW
        if m.dim() == 0:
            m = m.view(1, 1, 1)
            s = s.view(1, 1, 1)
        else:
            m = m.view(-1, 1, 1)
            s = s.view(-1, 1, 1)
    elif t.dim() == 4:
        # NCHW
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
    """Preprocess an image batch into a float32 tensor ready for CNNs.

    Steps
    -----
    1. Convert to ``torch.Tensor`` on ``device``.
    2. Reorder layout to channels-first (CHW/NCHW).
    3. Optionally convert RGB(A) → grayscale.
    4. Scaling:

       * If ``scale_uint8=True`` and dtype is integer, scale to [0, 1].
       * If dtype is float and values *appear* to be in 0..255 (max > ~1.5),
         scale by 1/255 as a defensive normalisation.

       After scaling, values are clamped to [0, 1] for stability.

    5. Apply optional mean/std normalisation.
    6. Return a tensor in CHW or NCHW (depending on input rank), float32.

    Parameters
    ----------
    x:
        Image or batch of images in HWC/NHWC/CHW/NCHW layouts; dtype uint8 or float.
    cfg:
        Preprocessing configuration; defaults are conservative (no mean/std shift).
    device:
        Target device for the output tensor.
    """
    cfg = cfg or ImagePreprocessConfig()

    t = as_tensor(x, device=torch.device(device))
    t = to_channels_first(t)  # -> CHW/NCHW

    if cfg.grayscale:
        # Only convert when we actually have RGB(A).
        ch_idx = 1 if t.dim() == 4 else 0
        if t.shape[ch_idx] in (3, 4):
            t = rgb_to_grayscale(t)

    # Scale integers to [0, 1]; and defensively scale float 0..255 inputs as well.
    t = _maybe_scale_uint8_or_float255(t, enable=cfg.scale_uint8)

    t = _apply_normalization(t, cfg.normalize_mean, cfg.normalize_std)

    if cfg.channels_first:
        return t  # already CHW/NCHW
    # Convert back to channels-last if requested.
    if t.dim() == 3:
        return t.permute(1, 2, 0)  # CHW -> HWC
    return t.permute(0, 2, 3, 1)  # NCHW -> NHWC
