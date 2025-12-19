from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irl.intrinsic import BaseIntrinsicModule, IntrinsicOutput, Transition
from irl.models import ConvEncoder, ConvEncoderConfig
from irl.utils.image import ImagePreprocessConfig, preprocess_image
from irl.utils.images import infer_channels_hw
from irl.utils.torchops import as_tensor, ensure_2d, one_hot

from .encoder import mlp
from .heads import ContinuousInverseHead, ForwardHead


@dataclass
class ICMConfig:
    phi_dim: int = 128
    hidden: Tuple[int, int] = (256, 256)
    lr: float = 3e-4
    beta_forward: float = 1.0
    beta_inverse: float = 1.0
    grad_clip: float = 5.0
    inv_log_std_min: float = -5.0
    inv_log_std_max: float = 2.0


class ICM(BaseIntrinsicModule, nn.Module):
    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        device: Union[str, torch.device] = "cpu",
        cfg: Optional[ICMConfig] = None,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("ICM supports Box observation spaces (vector or image).")
        self.device = torch.device(device)
        self.cfg = cfg or ICMConfig()

        self.is_image_obs = len(obs_space.shape) >= 2
        self.obs_dim = int(np.prod(obs_space.shape))

        self.is_discrete = isinstance(act_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.n_actions = int(act_space.n)
            self.act_dim = 1
            self._forward_act_in_dim = self.n_actions
        elif isinstance(act_space, gym.spaces.Box):
            self.n_actions = None
            self.act_dim = int(act_space.shape[0])
            self._forward_act_in_dim = self.act_dim
        else:
            raise TypeError(f"Unsupported action space for ICM: {type(act_space)}")

        if self.is_image_obs:
            shape = tuple(int(s) for s in obs_space.shape)
            in_channels, in_hw = infer_channels_hw(shape)

            cnn_cfg = ConvEncoderConfig(in_channels=int(in_channels), out_dim=int(self.cfg.phi_dim))
            self.encoder = ConvEncoder(cnn_cfg, in_hw=in_hw)
            self._img_pre_cfg = ImagePreprocessConfig(
                grayscale=False,
                scale_uint8=True,
                normalize_mean=None,
                normalize_std=None,
                channels_first=True,
            )
        else:
            self.encoder = mlp(int(obs_space.shape[0]), self.cfg.hidden, out_dim=self.cfg.phi_dim)

        inv_in = 2 * self.cfg.phi_dim
        if self.is_discrete:
            self.inverse = mlp(inv_in, self.cfg.hidden, out_dim=self.n_actions)
        else:
            self.inverse = ContinuousInverseHead(
                inv_in,
                self.act_dim,
                hidden=self.cfg.hidden,
                log_std_min=self.cfg.inv_log_std_min,
                log_std_max=self.cfg.inv_log_std_max,
            )

        fwd_in = self.cfg.phi_dim + self._forward_act_in_dim
        self.forward_head = ForwardHead(fwd_in, self.cfg.phi_dim, hidden=self.cfg.hidden)

        self._opt = torch.optim.Adam(self.parameters(), lr=float(self.cfg.lr))
        self.to(self.device)

    def _phi(self, obs: Any) -> Tensor:
        if self.is_image_obs:
            t = obs if torch.is_tensor(obs) else torch.as_tensor(obs)
            if t.dim() >= 5:
                t = t.reshape(-1, *t.shape[-3:])

            if (
                t.dim() == 3
                and int(self.encoder.in_channels) == 1
                and int(t.shape[-1]) not in (1, 3, 4)
            ):
                t = t.unsqueeze(0) if int(t.shape[0]) == 1 else t.unsqueeze(1)

            x = preprocess_image(t, cfg=self._img_pre_cfg, device=self.device)
            return self.encoder(x)
        x = as_tensor(obs, self.device)
        return self.encoder(ensure_2d(x))

    def _one_hot(self, a: Tensor, n: int) -> Tensor:
        return one_hot(a, n)

    def _act_for_forward(self, actions: Tensor) -> Tensor:
        if self.is_discrete:
            return self._one_hot(actions, self.n_actions)
        return ensure_2d(actions).float()

    def compute(self, tr: Transition) -> IntrinsicOutput:
        with torch.no_grad():
            r = self.compute_batch(tr.s, tr.s_next, tr.a, reduction="none")
            return IntrinsicOutput(r_int=float(r.view(-1)[0].item()))

    def compute_batch(
        self, obs: Any, next_obs: Any, actions: Any, reduction: str = "none"
    ) -> Tensor:
        with torch.no_grad():
            a = torch.as_tensor(actions, device=self.device)
            phi_t = self._phi(obs)
            phi_tp1 = self._phi(next_obs)
            a_fwd = self._act_for_forward(a)

            pred = self.forward_head(torch.cat([phi_t, a_fwd], dim=-1))
            mse_per = F.mse_loss(pred, phi_tp1, reduction="none").mean(dim=-1)
            if reduction == "mean":
                return mse_per.mean()
            return mse_per

    def _inverse_loss(self, phi_t: Tensor, phi_tp1: Tensor, actions: Tensor) -> Tensor:
        h = torch.cat([phi_t, phi_tp1], dim=-1)
        if self.is_discrete:
            logits = self.inverse(h)
            a = actions.long().view(-1)
            return F.cross_entropy(logits, a)

        mu, log_std = self.inverse(h)
        a = ensure_2d(actions).float()
        var = torch.exp(2.0 * log_std)
        nll = 0.5 * ((a - mu) ** 2 / var + 2.0 * log_std + np.log(2 * np.pi))
        return nll.sum(dim=-1).mean()

    def _forward_loss(self, phi_t: Tensor, actions_fwd: Tensor, phi_tp1: Tensor) -> Tensor:
        pred = self.forward_head(torch.cat([phi_t, actions_fwd], dim=-1))
        return F.mse_loss(pred, phi_tp1, reduction="mean")

    def loss(
        self,
        obs: Any,
        next_obs: Any,
        actions: Any,
        weights: Optional[Tuple[float, float]] = None,
    ) -> Mapping[str, Tensor]:
        a = torch.as_tensor(actions, device=self.device)

        phi_t = self._phi(obs)
        phi_tp1 = self._phi(next_obs)
        a_fwd = self._act_for_forward(a)

        loss_inv = self._inverse_loss(phi_t, phi_tp1, a)
        loss_fwd = self._forward_loss(phi_t, a_fwd, phi_tp1)

        beta_fwd, beta_inv = (
            (self.cfg.beta_forward, self.cfg.beta_inverse)
            if weights is None
            else (float(weights[0]), float(weights[1]))
        )
        total = beta_fwd * loss_fwd + beta_inv * loss_inv

        with torch.no_grad():
            r_int_mean = (
                F.mse_loss(
                    self.forward_head(torch.cat([phi_t, a_fwd], dim=-1)),
                    phi_tp1,
                    reduction="none",
                )
                .mean(dim=-1)
                .mean()
            )

        return {
            "total": total,
            "forward": loss_fwd,
            "inverse": loss_inv,
            "intrinsic_mean": r_int_mean,
        }

    def update(
        self,
        obs: Any,
        next_obs: Any,
        actions: Any,
        steps: int = 1,
        weights: Optional[Tuple[float, float]] = None,
    ) -> Mapping[str, float]:
        metrics: Mapping[str, float] = {}
        for _ in range(int(steps)):
            out = self.loss(obs, next_obs, actions, weights=weights)
            self._opt.zero_grad(set_to_none=True)
            out["total"].backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=float(self.cfg.grad_clip))
            self._opt.step()

            metrics = {
                "loss_total": float(out["total"].detach().item()),
                "loss_forward": float(out["forward"].detach().item()),
                "loss_inverse": float(out["inverse"].detach().item()),
                "intrinsic_mean": float(out["intrinsic_mean"].detach().item()),
            }
        return metrics
