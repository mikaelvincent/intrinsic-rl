from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from irl.utils.torchops import as_tensor

__all__ = ["ConvEncoder", "ConvEncoderConfig"]


@dataclass(frozen=True)
class ConvEncoderConfig:
    in_channels: int = 3
    channels: Tuple[int, int, int] = (32, 64, 64)
    kernels: Tuple[int, int, int] = (8, 4, 3)
    strides: Tuple[int, int, int] = (4, 2, 1)
    paddings: Tuple[int, int, int] = (0, 0, 1)
    out_dim: int = 256
    activation_inplace: bool = True


class ConvEncoder(nn.Module):
    def __init__(
        self,
        cfg: Optional[ConvEncoderConfig] = None,
        *,
        in_hw: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.cfg = cfg or ConvEncoderConfig()
        self.in_channels = int(self.cfg.in_channels)
        self.out_dim = int(self.cfg.out_dim)

        c = tuple(int(x) for x in self.cfg.channels)
        k = tuple(int(x) for x in self.cfg.kernels)
        s = tuple(int(x) for x in self.cfg.strides)
        p = tuple(int(x) for x in self.cfg.paddings)
        assert len(c) == len(k) == len(s) == len(p), (
            "channels/kernels/strides/paddings must match length"
        )

        layers: list[nn.Module] = []
        cin = self.in_channels
        for cout, kw, st, pad in zip(c, k, s, p):
            layers.append(nn.Conv2d(cin, cout, kernel_size=kw, stride=st, padding=pad))
            layers.append(nn.ReLU(inplace=bool(self.cfg.activation_inplace)))
            cin = cout
        self.conv = nn.Sequential(*layers)

        self._proj: Optional[nn.Linear] = None
        if in_hw is not None:
            self._initialize_projection(in_hw)

    def _initialize_projection(self, in_hw: Tuple[int, int]) -> None:
        with torch.no_grad():
            device = next(self.parameters()).device
            dummy = torch.zeros(1, self.in_channels, int(in_hw[0]), int(in_hw[1]), device=device)
            h = self.conv(dummy)
            flat = int(h.numel())
        self._proj = nn.Linear(flat, self.out_dim).to(device)

    @staticmethod
    def _ensure_nchw(x: Tensor, expected_c: int) -> Tensor:
        if x.dim() == 3:
            c_like = x.shape[0] in (1, 3, 4)
            if c_like and x.shape[0] == expected_c:
                x = x.unsqueeze(0)
            else:
                x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.dim() == 4:
            if x.shape[1] not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
                x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"ConvEncoder expects 3D/4D input, got {x.shape}")
        return x

    def forward(self, x: Union[Tensor, "numpy.ndarray"]) -> Tensor:
        device = next(self.parameters()).device
        t = as_tensor(x, device, dtype=torch.float32)
        t = self._ensure_nchw(t, self.in_channels)

        h = self.conv(t)
        h = h.reshape(h.size(0), -1)

        if self._proj is None:
            self._proj = nn.Linear(h.size(1), self.out_dim).to(h.device)

        return self._proj(h)
