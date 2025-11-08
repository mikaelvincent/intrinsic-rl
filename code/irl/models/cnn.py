from __future__ import annotations

"""Lightweight CNN encoder for image observations.

This module adds a configurable convolutional encoder that maps images
(NCHW/NHWC; grayscale or RGB) to a fixed-size feature vector, intended for
use by image-based agents (e.g., CarRacing) in later integration steps.

Default architecture follows the spec (Sprint 6 §6): three conv blocks
with strides (4, 2, 1) and a linear projection to 256 features.

Usage
-----
>>> enc = ConvEncoder(in_channels=3, out_dim=256)
>>> x = torch.rand(8, 3, 96, 96)  # NCHW floats in [0, 1]
>>> z = enc(x)  # [8, 256]

The encoder is robust to NHWC inputs and will lazily create its final
projection layer on the first forward pass if the input spatial size is
not known at construction time.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from irl.utils.torchops import as_tensor


__all__ = ["ConvEncoder", "ConvEncoderConfig"]


@dataclass(frozen=True)
class ConvEncoderConfig:
    """Configuration for the ConvEncoder."""

    in_channels: int = 3
    channels: Tuple[int, int, int] = (32, 64, 64)
    kernels: Tuple[int, int, int] = (8, 4, 3)
    strides: Tuple[int, int, int] = (4, 2, 1)
    # Use padding on the final 3x3 layer to avoid underflow on small inputs (e.g., 32x32 -> 2x2).
    paddings: Tuple[int, int, int] = (0, 0, 1)
    out_dim: int = 256
    activation_inplace: bool = True


class ConvEncoder(nn.Module):
    """CNN feature extractor (configurable) → fixed-dim embedding.

    Expects float inputs in [0, 1] with shape [N, C, H, W] or [N, H, W, C].
    If the final projection size cannot be inferred at construction time,
    it is created lazily on the first forward pass.

    Parameters
    ----------
    cfg:
        ConvEncoderConfig with conv and projection settings.
    in_hw:
        Optional spatial size (H, W). If provided, the final projection
        layer is constructed immediately; otherwise it is initialized lazily.

    Notes
    -----
    * This module does not perform image preprocessing (scaling, grayscale).
      Use `irl.utils.image.preprocess_image` for that.
    """

    def __init__(self, cfg: Optional[ConvEncoderConfig] = None, *, in_hw: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.cfg = cfg or ConvEncoderConfig()
        self.in_channels = int(self.cfg.in_channels)
        self.out_dim = int(self.cfg.out_dim)

        c = tuple(int(x) for x in self.cfg.channels)
        k = tuple(int(x) for x in self.cfg.kernels)
        s = tuple(int(x) for x in self.cfg.strides)
        p = tuple(int(x) for x in self.cfg.paddings)

        assert len(c) == len(k) == len(s) == len(p), "channels/kernels/strides/paddings must match length"

        layers: list[nn.Module] = []
        cin = self.in_channels
        for cout, kw, st, pad in zip(c, k, s, p):
            layers.append(nn.Conv2d(cin, cout, kernel_size=kw, stride=st, padding=pad))
            layers.append(nn.ReLU(inplace=bool(self.cfg.activation_inplace)))
            cin = cout
        self.conv = nn.Sequential(*layers)

        # Final projection is built lazily unless in_hw is known
        self._proj: Optional[nn.Linear] = None
        if in_hw is not None:
            self._initialize_projection(in_hw)

    # ----------------------- helpers -----------------------

    def _initialize_projection(self, in_hw: Tuple[int, int]) -> None:
        """Materialize the final Linear based on (H, W)."""
        with torch.no_grad():
            device = next(self.parameters()).device
            dummy = torch.zeros(1, self.in_channels, int(in_hw[0]), int(in_hw[1]), device=device)
            h = self.conv(dummy)
            flat = int(h.numel())
        # Create on the same device as the conv stack
        self._proj = nn.Linear(flat, self.out_dim).to(device)

    @staticmethod
    def _ensure_nchw(x: Tensor, expected_c: int) -> Tensor:
        """Return a NCHW tensor given NCHW/NHWC or CHW/HWC inputs."""
        if x.dim() == 3:
            # CHW or HWC
            c_like = x.shape[0] in (1, 3, 4)
            if c_like and x.shape[0] == expected_c:
                x = x.unsqueeze(0)  # -> NCHW
            else:
                # Assume HWC
                x = x.permute(2, 0, 1).unsqueeze(0)  # -> NCHW
        elif x.dim() == 4:
            # NCHW or NHWC
            if x.shape[1] not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
                x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        else:
            raise ValueError(f"ConvEncoder expects 3D/4D input, got {x.shape}")
        return x

    # ----------------------- forward -----------------------

    def forward(self, x: Union[Tensor, "numpy.ndarray"]) -> Tensor:  # noqa: F821 - numpy is optional at runtime
        device = next(self.parameters()).device
        t = as_tensor(x, device, dtype=torch.float32)
        t = self._ensure_nchw(t, self.in_channels)

        h = self.conv(t)
        # Use reshape to support non-contiguous tensors safely.
        h = h.reshape(h.size(0), -1)  # [N, F]

        if self._proj is None:
            # Lazily create projection with correct input size, on the right device
            self._proj = nn.Linear(h.size(1), self.out_dim).to(h.device)

        z = self._proj(h)
        return z
