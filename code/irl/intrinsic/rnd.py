from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Union

import gymnasium as gym
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irl.intrinsic import BaseIntrinsicModule, IntrinsicOutput
from irl.models.cnn import ConvEncoder, ConvEncoderConfig
from irl.models.networks import mlp
from irl.utils.image import ImagePreprocessConfig, preprocess_image
from irl.utils.torchops import as_tensor, ensure_2d


@dataclass
class RNDConfig:
    feature_dim: int = 128
    hidden: Tuple[int, int] = (256, 256)
    lr: float = 3e-4
    grad_clip: float = 5.0
    rms_beta: float = 0.99
    rms_eps: float = 1e-8
    normalize_intrinsic: bool = False


class RND(BaseIntrinsicModule, nn.Module):
    # outputs_normalized tells the trainer whether it should apply its global RMS.
    def __init__(
        self,
        obs_space: gym.Space,
        device: Union[str, torch.device] = "cpu",
        cfg: Optional[RNDConfig] = None,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("RND supports Box observation spaces (vector or images).")

        self.device = torch.device(device)
        self.cfg = cfg or RNDConfig()
        self.is_image = len(obs_space.shape) >= 2

        if self.is_image:
            shape = tuple(int(s) for s in obs_space.shape)
            if len(shape) == 3:
                c0 = shape[0]
                c2 = shape[-1]
                if c0 in (1, 3, 4) and c2 not in (1, 3, 4):
                    in_channels, in_hw = c0, (shape[1], shape[2])
                else:
                    in_channels, in_hw = c2, (shape[0], shape[1])
            else:
                in_channels = shape[-1]
                in_hw = (shape[0], shape[1])

            cnn_cfg = ConvEncoderConfig(
                in_channels=int(in_channels), out_dim=int(self.cfg.feature_dim)
            )
            self.target = ConvEncoder(cnn_cfg, in_hw=in_hw)
            self.predictor = ConvEncoder(cnn_cfg, in_hw=in_hw)

            for p in self.target.parameters():
                p.requires_grad = False
            self.target.eval()

            self._img_pre_cfg = ImagePreprocessConfig(
                grayscale=False,
                scale_uint8=True,
                normalize_mean=None,
                normalize_std=None,
                channels_first=True,
            )
        else:
            self.obs_dim = int(obs_space.shape[0])
            self.target = mlp(self.obs_dim, self.cfg.hidden, out_dim=self.cfg.feature_dim)
            self.predictor = mlp(self.obs_dim, self.cfg.hidden, out_dim=self.cfg.feature_dim)
            for p in self.target.parameters():
                p.requires_grad = False
            self.target.eval()

        self._opt = torch.optim.Adam(self.predictor.parameters(), lr=float(self.cfg.lr))

        # Start non-zero to avoid tiny denominators early.
        self.register_buffer("_r2_ema", torch.tensor(1.0, dtype=torch.float32))

        self.outputs_normalized: bool = bool(self.cfg.normalize_intrinsic)
        self.to(self.device)

    @property
    def rms(self) -> float:
        return float(torch.sqrt(self._r2_ema + self.cfg.rms_eps).detach().item())

    def _pred_and_targ(self, x: Tensor | object) -> Tuple[Tensor, Tensor]:
        if self.is_image:
            t = x if torch.is_tensor(x) else torch.as_tensor(x)
            if t.dim() >= 5:
                t = t.reshape(-1, *t.shape[-3:])
            img = preprocess_image(t, cfg=self._img_pre_cfg, device=self.device)
            p = self.predictor(img)
            with torch.no_grad():
                tgt = self.target(img)
            return p, tgt

        x2 = ensure_2d(as_tensor(x, self.device))
        p = self.predictor(x2)
        with torch.no_grad():
            tgt = self.target(x2)
        return p, tgt

    def _intrinsic_raw_per_sample(self, x: Tensor | object) -> Tensor:
        p, tgt = self._pred_and_targ(x)
        return F.mse_loss(p, tgt, reduction="none").mean(dim=-1)

    def _update_rms_from_raw(self, per: Tensor) -> None:
        if per.numel() == 0:
            return
        with torch.no_grad():
            vals = per.to(device=self._r2_ema.device, dtype=torch.float32)
            r2_batch_mean = (vals**2).mean()
            self._r2_ema.mul_(self.cfg.rms_beta).add_((1.0 - self.cfg.rms_beta) * r2_batch_mean)

    def _normalize_intrinsic(self, r: Tensor) -> Tensor:
        if not self.cfg.normalize_intrinsic:
            return r
        denom = torch.sqrt(self._r2_ema + self.cfg.rms_eps)
        return r / denom

    def compute(self, tr) -> IntrinsicOutput:
        with torch.no_grad():
            x = tr.s_next if hasattr(tr, "s_next") and tr.s_next is not None else tr.s
            r_raw = self._intrinsic_raw_per_sample(x)
            if self.cfg.normalize_intrinsic:
                self._update_rms_from_raw(r_raw)
            r = self._normalize_intrinsic(r_raw)
            return IntrinsicOutput(r_int=float(r.view(-1)[0].item()))

    def compute_batch(
        self, obs: Any, next_obs: Any | None = None, reduction: str = "none"
    ) -> Tensor:
        with torch.no_grad():
            x_src = next_obs if next_obs is not None else obs
            r_raw = self._intrinsic_raw_per_sample(x_src)
            if self.cfg.normalize_intrinsic:
                self._update_rms_from_raw(r_raw)
            r = self._normalize_intrinsic(r_raw)
            return r.mean() if reduction == "mean" else r

    def loss(self, obs: Any) -> Mapping[str, Tensor]:
        p, tgt = self._pred_and_targ(obs)
        per = F.mse_loss(p, tgt, reduction="none").mean(dim=-1)
        total = per.mean()

        # Keep RMS fresh for diagnostics when trainer normalizes globally.
        if not self.cfg.normalize_intrinsic:
            self._update_rms_from_raw(per)

        return {"total": total, "intrinsic_mean": per.mean()}

    def update(self, obs: Any, steps: int = 1) -> Mapping[str, float]:
        metrics: Mapping[str, float] = {}
        for _ in range(int(steps)):
            out = self.loss(obs)
            self._opt.zero_grad(set_to_none=True)
            out["total"].backward()
            nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=float(self.cfg.grad_clip))
            self._opt.step()

            metrics = {
                "loss_total": float(out["total"].detach().item()),
                "loss_intrinsic_mean": float(out["intrinsic_mean"].detach().item()),
                "rms": self.rms,
            }
        return metrics
