"""ICM intrinsic module (split into encoder/heads/module).

Implements:
- Representation encoder φ(s)
- Inverse head (CE for Discrete, Gaussian NLL for Box)
- Forward head (MSE in φ-space)

Public API:
    * ICMConfig
    * ICM (compute / compute_batch / loss / update)
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
    """Intrinsic Curiosity Module with φ, inverse, and forward heads."""

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        device: Union[str, torch.device] = "cpu",
        cfg: Optional[ICMConfig] = None,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("ICM supports Box observation spaces (vector).")
        self.device = torch.device(device)
        self.cfg = cfg or ICMConfig()

        # ----- Spaces -----
        self.obs_dim = int(obs_space.shape[0])
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

        # ----- Networks -----
        # Encoder φ(s) : R^obs_dim -> R^phi_dim
        self.encoder = mlp(self.obs_dim, self.cfg.hidden, out_dim=self.cfg.phi_dim)

        # Inverse dynamics head: input = concat[φ(s), φ(s')]
        inv_in = 2 * self.cfg.phi_dim
        if self.is_discrete:
            self.inverse = mlp(inv_in, self.cfg.hidden, out_dim=self.n_actions)
        else:
            self.inverse = ContinuousInverseHead(inv_in, self.act_dim, hidden=self.cfg.hidden)

        # Forward dynamics head: input = concat[φ(s), a_embed] -> φ̂(s')
        fwd_in = self.cfg.phi_dim + self._forward_act_in_dim
        self.forward_head = ForwardHead(fwd_in, self.cfg.phi_dim, hidden=self.cfg.hidden)

        # Optimizer
        self._opt = torch.optim.Adam(self.parameters(), lr=float(self.cfg.lr))

        self.to(self.device)

    # ----------------------------- Embeddings -----------------------------

    def _phi(self, obs: Tensor) -> Tensor:
        x = ensure_2d(obs)
        return self.encoder(x)

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
            s = as_tensor(tr.s, self.device)
            s_next = as_tensor(tr.s_next, self.device)
            a = as_tensor(tr.a, self.device)

            r = self.compute_batch(
                s.view(1, -1),
                s_next.view(1, -1),
                a.view(1, -1 if not self.is_discrete else 1),
            )
            return IntrinsicOutput(r_int=float(r.view(-1)[0].item()))

    def compute_batch(
        self, obs: Any, next_obs: Any, actions: Any, reduction: str = "none"
    ) -> Tensor:
        """Vectorized intrinsic reward: per-sample forward MSE in φ-space."""
        with torch.no_grad():
            o = as_tensor(obs, self.device)
            op = as_tensor(next_obs, self.device)
            a = as_tensor(actions, self.device)
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
        o = as_tensor(obs, self.device)
        op = as_tensor(next_obs, self.device)
        a = as_tensor(actions, self.device)

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
