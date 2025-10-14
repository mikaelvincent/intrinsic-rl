"""ICM intrinsic module.

Implements the "Intrinsic Curiosity Module" with:
- Representation encoder φ(s)
- Inverse dynamics head: predicts action a | φ(s), φ(s')
  * Discrete actions: cross-entropy loss
  * Continuous (Box) actions: diagonal Gaussian NLL (mean/log_std predicted)
- Forward dynamics head: predicts next representation φ̂(s') | φ(s), a
  * Loss: MSE in representation space

Intrinsic reward (per transition): forward-prediction MSE (no η scaling here).
Scale by `cfg.intrinsic.eta` in the trainer when combining with extrinsic reward.

Notes
-----
* Supports vector observation spaces (gym.spaces.Box) only (images/CNN planned).
* Handles both Discrete and Box action spaces for inverse/forward heads.
* Provides:
    - compute(tr): Transition -> IntrinsicOutput(r_int=float)
    - compute_batch(...): vectorized intrinsic for (obs, next_obs, actions)
    - loss(...): total, forward, inverse losses (no optimizer step)
    - update(...): performs optimization step(s), returns metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import BaseIntrinsicModule, IntrinsicOutput, Transition


# ------------------------------ Small helpers ------------------------------


def _as_tensor(x: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype or torch.float32)


def _ensure_2d(x: Tensor) -> Tensor:
    """Ensure [B, D]; if [D], add batch dim; if [T,B,D] flatten to [T*B, D]."""
    if x.dim() == 1:
        return x.view(1, -1)
    if x.dim() == 2:
        return x
    # Flatten all but last dim
    return x.view(-1, x.size(-1))


def _mlp(in_dim: int, hidden: Iterable[int], out_dim: Optional[int] = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
        last = h
    if out_dim is not None:
        layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


# ------------------------------ ICM Networks -------------------------------


class _ICMContinuousInvHead(nn.Module):
    """Inverse head for continuous actions: predicts mean and (bounded) log_std."""

    def __init__(self, in_dim: int, act_dim: int, hidden: Iterable[int] = (256, 256)) -> None:
        super().__init__()
        self.backbone = _mlp(in_dim, hidden, out_dim=2 * act_dim)
        # clamp range for numerical stability
        self.log_std_min = -5.0
        self.log_std_max = 2.0
        self.act_dim = act_dim

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.backbone(x)  # [B, 2*A]
        mu, log_std = torch.split(h, self.act_dim, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std


class _ICMForward(nn.Module):
    """Forward dynamics head predicting next embedding φ̂(s')."""

    def __init__(self, in_dim: int, out_dim: int, hidden: Iterable[int] = (256, 256)) -> None:
        super().__init__()
        self.net = _mlp(in_dim, hidden, out_dim=out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------- ICM ------------------------------------


@dataclass
class ICMConfig:
    """Lightweight, ICM-local configuration.

    The repository's global config schema deliberately avoids method-specific fields at Sprint 1. These knobs are kept
    local (defaults mirror the spec).
    """

    phi_dim: int = 128
    hidden: Tuple[int, int] = (256, 256)
    lr: float = 3e-4
    beta_forward: float = 1.0  # weight for forward MSE in total loss
    beta_inverse: float = 1.0  # weight for inverse loss in total loss
    grad_clip: float = 5.0


class ICM(BaseIntrinsicModule, nn.Module):
    """Trainable Intrinsic Curiosity Module.

    Args:
        obs_space: gym.spaces.Box observation space (vector observations).
        act_space: Discrete or Box action space.
        device: torch device string or device.
        cfg: ICMConfig (local defaults suitable for Sprint 1).

    API:
        * compute(tr) -> IntrinsicOutput
        * compute_batch(obs, next_obs, actions) -> Tensor [B] intrinsic
        * loss(obs, next_obs, actions) -> dict
        * update(obs, next_obs, actions, steps=1) -> dict
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
            raise TypeError("ICM currently supports Box observation spaces (vector).")
        self.device = torch.device(device)
        self.cfg = cfg or ICMConfig()

        # ----- Spaces -----
        self.obs_dim = int(obs_space.shape[0])
        self.is_discrete = isinstance(act_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.n_actions = int(act_space.n)
            self.act_dim = 1  # for typing; one int index per sample
            self._forward_act_in_dim = self.n_actions  # one-hot for forward
        elif isinstance(act_space, gym.spaces.Box):
            self.n_actions = None
            self.act_dim = int(act_space.shape[0])
            self._forward_act_in_dim = self.act_dim
        else:
            raise TypeError(f"Unsupported action space for ICM: {type(act_space)}")

        # ----- Networks -----
        # Encoder φ(s) : R^obs_dim -> R^phi_dim
        self.encoder = _mlp(self.obs_dim, self.cfg.hidden, out_dim=self.cfg.phi_dim)

        # Inverse dynamics head: input = concat[φ(s), φ(s')]
        inv_in = 2 * self.cfg.phi_dim
        if self.is_discrete:
            # Direct logits for discrete a
            self.inverse = _mlp(inv_in, self.cfg.hidden, out_dim=self.n_actions)
        else:
            # Predict mean and log_std for continuous a
            self.inverse = _ICMContinuousInvHead(inv_in, self.act_dim, hidden=self.cfg.hidden)

        # Forward dynamics head: input = concat[φ(s), a_embed] -> φ̂(s')
        fwd_in = self.cfg.phi_dim + self._forward_act_in_dim
        self.forward_head = _ICMForward(fwd_in, self.cfg.phi_dim, hidden=self.cfg.hidden)

        # Optimizer
        self._opt = torch.optim.Adam(self.parameters(), lr=float(self.cfg.lr))

        # Move to device
        self.to(self.device)

    # ----------------------------- Embeddings -----------------------------

    def _phi(self, obs: Tensor) -> Tensor:
        x = _ensure_2d(obs)
        return self.encoder(x)

    # ----------------------- Action formatting helpers --------------------

    def _one_hot(self, a: Tensor, n: int) -> Tensor:
        a = a.long().view(-1)  # [B]
        oh = torch.zeros((a.size(0), n), device=a.device, dtype=torch.float32)
        oh.scatter_(1, a.view(-1, 1), 1.0)
        return oh

    def _act_for_forward(self, actions: Tensor) -> Tensor:
        """Format actions for the forward model input."""
        if self.is_discrete:
            return self._one_hot(actions, self.n_actions)  # [B, A]
        return _ensure_2d(actions).float()  # [B, A]

    # -------------------------- Intrinsic compute -------------------------

    def compute(self, tr: Transition) -> IntrinsicOutput:
        """Compute intrinsic reward for a single transition (no gradients)."""
        with torch.no_grad():
            s = _as_tensor(tr.s, self.device)
            s_next = _as_tensor(tr.s_next, self.device)
            a = _as_tensor(tr.a, self.device)

            r = self.compute_batch(s.view(1, -1), s_next.view(1, -1), a.view(1, -1 if not self.is_discrete else 1))
            # r: [1] tensor
            return IntrinsicOutput(r_int=float(r.item()))

    def compute_batch(
        self, obs: Any, next_obs: Any, actions: Any, reduction: str = "none"
    ) -> Tensor:
        """Vectorized intrinsic reward for batches.

        Returns:
            Tensor of shape [B] if reduction=="none"; scalar [1] if "mean".
        """
        with torch.no_grad():
            o = _as_tensor(obs, self.device)
            op = _as_tensor(next_obs, self.device)
            a = _as_tensor(actions, self.device)
            phi_t = self._phi(o)
            phi_tp1 = self._phi(op)
            a_fwd = self._act_for_forward(a)

            # Predict φ̂(s')
            pred = self.forward_head(torch.cat([phi_t, a_fwd], dim=-1))
            # Per-sample forward MSE over embedding dims
            mse_per = F.mse_loss(pred, phi_tp1, reduction="none").mean(dim=-1)  # [B]
            if reduction == "mean":
                return mse_per.mean()
            return mse_per

    # ----------------------------- Losses/Update ---------------------------

    def _inverse_loss(self, phi_t: Tensor, phi_tp1: Tensor, actions: Tensor) -> Tensor:
        h = torch.cat([phi_t, phi_tp1], dim=-1)
        if self.is_discrete:
            logits = self.inverse(h)  # [B, n_actions]
            a = actions.long().view(-1)  # [B]
            return F.cross_entropy(logits, a)
        # continuous
        mu, log_std = self.inverse(h)  # [B, act_dim] each
        a = _ensure_2d(actions).float()
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
        """Compute ICM losses without updating parameters.

        Returns dict with keys: total, forward, inverse, intrinsic_mean
        """
        o = _as_tensor(obs, self.device)
        op = _as_tensor(next_obs, self.device)
        a = _as_tensor(actions, self.device)

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

        # Intrinsic reward mean for diagnostics (no η)
        with torch.no_grad():
            r_int_mean = F.mse_loss(
                self.forward_head(torch.cat([phi_t, a_fwd], dim=-1)),
                phi_tp1,
                reduction="none",
            ).mean(dim=-1).mean()

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
        """Run optimization for `steps` iterations on the same batch.

        Returns scalar metrics (floats).
        """
        metrics = {}
        for _ in range(int(steps)):
            out = self.loss(obs, next_obs, actions, weights=weights)
            self._opt.zero_grad(set_to_none=True)
            out["total"].backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=float(self.cfg.grad_clip))
            self._opt.step()
            # Accumulate last values
            metrics = {
                "loss_total": float(out["total"].detach().item()),
                "loss_forward": float(out["forward"].detach().item()),
                "loss_inverse": float(out["inverse"].detach().item()),
                "intrinsic_mean": float(out["intrinsic_mean"].detach().item()),
            }
        return metrics
