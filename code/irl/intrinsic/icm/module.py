"""ICM intrinsic module: encoder, inverse, and forward heads.

Intrinsic = per-sample forward MSE in φ-space. Includes compute, loss, and update
routines for both discrete and continuous actions. Now supports **image**
observations by routing through a ConvEncoder when the observation space has
rank ≥ 2 (HWC or CHW). See devspec §5.3.2 and §6 (Sprint 6).

Image dtype contract
--------------------
This module accepts **raw uint8 images** (HWC/NHWC/CHW/NCHW) or **float images already
in [0, 1]**. If floats appear to be in [0, 255], the shared preprocessing pipeline
(`utils.image.preprocess_image`) will defensively scale them to [0, 1]. To avoid
accidentally bypassing scaling, do **not** pre-cast images to float32 unless you also
scale to [0, 1]; instead, pass the original arrays/tensors and let preprocessing handle
layout and normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irl.intrinsic import BaseIntrinsicModule, IntrinsicOutput, Transition
from irl.utils.torchops import as_tensor, ensure_2d, one_hot
from irl.models import ConvEncoder, ConvEncoderConfig  # CNN path for image observations
from irl.utils.image import preprocess_image, ImagePreprocessConfig

from .encoder import mlp
from .heads import ContinuousInverseHead, ForwardHead


# ------------------------------ Config ----------------------------------


@dataclass
class ICMConfig:
    """Lightweight, ICM-local configuration."""

    phi_dim: int = 128
    hidden: Tuple[int, int] = (256, 256)
    lr: float = 3e-4
    beta_forward: float = 1.0  # weight for forward MSE in total loss
    beta_inverse: float = 1.0  # weight for inverse loss in total loss
    grad_clip: float = 5.0


# ---------------------------------- ICM ------------------------------------


class ICM(BaseIntrinsicModule, nn.Module):
    """Intrinsic Curiosity Module with φ, inverse, and forward heads.

    * Vector observations → φ via an MLP.
    * Image observations (rank ≥ 2) → φ via ConvEncoder (CHW/NCHW/NHWC friendly) with centralized preprocessing.
    """

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

        # ----- Spaces -----
        self.is_image_obs = len(obs_space.shape) >= 2  # HWC/CHW or higher rank
        # For vectors this is the feature dimension; for images it's unused by encoder logic
        self.obs_dim = (
            int(np.prod(obs_space.shape))
            if not self.is_image_obs
            else int(np.prod(obs_space.shape))
        )

        self.is_discrete = isinstance(act_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.n_actions = int(act_space.n)
            self.act_dim = 1
            self._forward_act_in_dim = self.n_actions  # one-hot for forward
        elif isinstance(act_space, gym.spaces.Box):
            self.n_actions = None
            self.act_dim = int(act_space.shape[0])
            self._forward_act_in_dim = self.act_dim
        else:
            raise TypeError(f"Unsupported action space for ICM: {type(act_space)}")

        # ----- Encoder φ(s) -----
        if self.is_image_obs:
            # Infer channel count from either leading or trailing axis (CHW vs HWC)
            shape = tuple(int(s) for s in obs_space.shape)
            cand = [shape[0], shape[-1]]
            in_channels = cand[0] if cand[0] in (1, 3, 4) else cand[1]
            if in_channels not in (1, 3, 4):
                # Fallback: assume channels-last
                in_channels = shape[-1]
            cnn_cfg = ConvEncoderConfig(in_channels=int(in_channels), out_dim=int(self.cfg.phi_dim))
            self.encoder = ConvEncoder(cnn_cfg)  # lazy projection init handled internally

            # Centralized image preprocessing config: keep original channels; output NCHW; scale uint8 → [0,1]
            self._img_pre_cfg = ImagePreprocessConfig(
                grayscale=False,
                scale_uint8=True,
                normalize_mean=None,
                normalize_std=None,
                channels_first=True,
            )
        else:
            # Vector path: simple MLP
            # Note: first layer of mlp() includes FlattenObs so [T,B,D]/[B,D]/[D] are all handled.
            self.encoder = mlp(int(obs_space.shape[0]), self.cfg.hidden, out_dim=self.cfg.phi_dim)

        # ----- Inverse dynamics head: input = concat[φ(s), φ(s')] -----
        inv_in = 2 * self.cfg.phi_dim
        if self.is_discrete:
            self.inverse = mlp(inv_in, self.cfg.hidden, out_dim=self.n_actions)
        else:
            self.inverse = ContinuousInverseHead(inv_in, self.act_dim, hidden=self.cfg.hidden)

        # ----- Forward dynamics head: input = concat[φ(s), a_embed] -> φ̂(s') -----
        fwd_in = self.cfg.phi_dim + self._forward_act_in_dim
        self.forward_head = ForwardHead(fwd_in, self.cfg.phi_dim, hidden=self.cfg.hidden)

        # Optimizer
        self._opt = torch.optim.Adam(self.parameters(), lr=float(self.cfg.lr))

        self.to(self.device)

    # ----------------------------- Embeddings -----------------------------

    def _phi(self, obs: Any) -> Tensor:
        """Return φ(s) for a batch or single observation.

        * If image: centralized preprocessing to NCHW float32 in [0,1] via `utils.image.preprocess_image`,
          tolerant to HWC/NHWC/CHW/NCHW and extra leading dims (e.g., (T,B,H,W,C)).
        * If vector: flatten to [B, D] before passing to the MLP encoder.
        """
        if self.is_image_obs:
            t = obs if torch.is_tensor(obs) else torch.as_tensor(obs)
            if t.dim() >= 5:
                # Collapse leading dims to a simple NCHW/NHWC batch
                t = t.reshape(-1, *t.shape[-3:])
            x = preprocess_image(
                t, cfg=self._img_pre_cfg, device=self.device
            )  # -> NCHW float32 [0,1]
            return self.encoder(x)
        else:
            x = as_tensor(obs, self.device)
            return self.encoder(ensure_2d(x))

    # ----------------------- Action formatting helpers --------------------

    def _one_hot(self, a: Tensor, n: int) -> Tensor:
        return one_hot(a, n)

    def _act_for_forward(self, actions: Tensor) -> Tensor:
        """Format actions for forward model input."""
        if self.is_discrete:
            return self._one_hot(actions, self.n_actions)  # [B, A]
        return ensure_2d(actions).float()  # [B, A]

    # -------------------------- Intrinsic compute -------------------------

    def compute(self, tr: Transition) -> IntrinsicOutput:
        """Compute intrinsic (per-sample forward MSE) without gradients."""
        with torch.no_grad():
            s = tr.s
            s_next = tr.s_next
            a = tr.a

            r = self.compute_batch(s, s_next, a, reduction="none")
            return IntrinsicOutput(r_int=float(r.view(-1)[0].item()))

    def compute_batch(
        self, obs: Any, next_obs: Any, actions: Any, reduction: str = "none"
    ) -> Tensor:
        """Vectorized intrinsic reward: per-sample forward MSE in φ-space."""
        with torch.no_grad():
            o = obs
            op = next_obs
            a = torch.as_tensor(actions, device=self.device)
            phi_t = self._phi(o)
            phi_tp1 = self._phi(op)
            a_fwd = self._act_for_forward(a)

            pred = self.forward_head(torch.cat([phi_t, a_fwd], dim=-1))
            mse_per = F.mse_loss(pred, phi_tp1, reduction="none").mean(dim=-1)  # [B]
            if reduction == "mean":
                return mse_per.mean()
            return mse_per

    # ----------------------------- Losses/Update ---------------------------

    def _inverse_loss(self, phi_t: Tensor, phi_tp1: Tensor, actions: Tensor) -> Tensor:
        h = torch.cat([phi_t, phi_tp1], dim=-1)
        if self.is_discrete:
            logits = self.inverse(h)  # [B, n_actions]
            a = actions.long().view(-1)
            return F.cross_entropy(logits, a)
        # continuous
        mu, log_std = self.inverse(h)  # type: ignore[misc]
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
        """Compute ICM losses without updating parameters."""
        o = obs
        op = next_obs
        a = torch.as_tensor(actions, device=self.device)

        phi_t = self._phi(o)
        phi_tp1 = self._phi(op)
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
        """Optimize predictor/encoder for `steps` iterations on the same batch."""
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
