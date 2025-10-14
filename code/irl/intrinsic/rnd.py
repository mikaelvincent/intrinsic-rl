"""RND intrinsic module.

Random Network Distillation (RND) computes intrinsic reward as the prediction error
of a trainable "predictor" network trying to match the output of a fixed, randomly
initialized "target" network on observations.

* Intrinsic reward per sample:  MSE(predictor(x), target(x))
  - (optionally normalized online by a running RMS; see config)

API (mirrors ICM for convenience)
---------------------------------
- compute(tr) -> IntrinsicOutput
- compute_batch(obs, next_obs=None, reduction="none") -> Tensor[[B] or [1]]
- loss(obs) -> dict(total, intrinsic_mean)
- update(obs, steps=1) -> dict(loss_total, intrinsic_mean)

Notes
-----
* This sprint assumes **vector observations** (gym.spaces.Box).
* Actions are not required by RND and therefore ignored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irl.intrinsic import BaseIntrinsicModule, IntrinsicOutput
from irl.models.networks import mlp  # lightweight MLP builder (+FlattenObs)


# ------------------------------ Helpers ---------------------------------


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
    return x.view(-1, x.size(-1))


# ------------------------------ Config ----------------------------------


@dataclass
class RNDConfig:
    """Lightweight configuration for RND (kept local to the module).

    Attributes
    ----------
    feature_dim : int
        Output dimensionality of target/predictor MLPs.
    hidden : Tuple[int, int]
        Hidden sizes for the MLPs.
    lr : float
        Learning rate for the predictor's Adam optimizer.
    grad_clip : float
        Global gradient-norm clipping value.
    rms_beta : float
        Exponential moving-average coefficient for running RMS normalization.
    rms_eps : float
        Epsilon for RMS denominator numerical stability.
    normalize_intrinsic : bool
        If True, returned intrinsic is divided by running RMS.
        NOTE: Default is False because the trainer now provides a global
        RMS normalizer for intrinsic rewards. You can re-enable locally if needed.
    """

    feature_dim: int = 128
    hidden: Tuple[int, int] = (256, 256)
    lr: float = 3e-4
    grad_clip: float = 5.0
    rms_beta: float = 0.99
    rms_eps: float = 1e-8
    normalize_intrinsic: bool = False  # global normalizer handles default scaling


# ------------------------------ Module ----------------------------------


class RND(BaseIntrinsicModule, nn.Module):
    """Random Network Distillation (target/predictor; intrinsic = MSE)."""

    def __init__(
        self,
        obs_space: gym.Space,
        device: Union[str, torch.device] = "cpu",
        cfg: Optional[RNDConfig] = None,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("RND currently supports Box observation spaces (vector states).")

        self.device = torch.device(device)
        self.cfg = cfg or RNDConfig()
        self.obs_dim = int(obs_space.shape[0])

        # Target: fixed, randomly initialized; Predictor: trainable
        self.target = mlp(self.obs_dim, self.cfg.hidden, out_dim=self.cfg.feature_dim)
        self.predictor = mlp(self.obs_dim, self.cfg.hidden, out_dim=self.cfg.feature_dim)

        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()  # just to be explicit

        self._opt = torch.optim.Adam(self.predictor.parameters(), lr=float(self.cfg.lr))

        # Running RMS over *unnormalized* per-sample intrinsic values
        # We track exponential moving average of r^2 to estimate RMS.
        self.register_buffer("_r2_ema", torch.tensor(1.0, dtype=torch.float32))  # start non-zero

        self.to(self.device)

    # -------------------------- Core compute ---------------------------

    def _pred_and_targ(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x2 = _ensure_2d(x)
        p = self.predictor(x2)
        with torch.no_grad():
            t = self.target(x2)
        return p, t

    def _intrinsic_raw_per_sample(self, x: Tensor) -> Tensor:
        """Per-sample (vector reduced) MSE, shape [B]."""
        p, t = self._pred_and_targ(x)
        return F.mse_loss(p, t, reduction="none").mean(dim=-1)

    def _normalize_intrinsic(self, r: Tensor) -> Tensor:
        if not self.cfg.normalize_intrinsic:
            return r
        denom = torch.sqrt(self._r2_ema + self.cfg.rms_eps)
        return r / denom

    # Public API

    def compute(self, tr) -> IntrinsicOutput:
        """Compute intrinsic for a single transition (no gradients)."""
        with torch.no_grad():
            x = tr.s_next if hasattr(tr, "s_next") and tr.s_next is not None else tr.s
            xt = _as_tensor(x, self.device)
            r = self._intrinsic_raw_per_sample(xt)
            r = self._normalize_intrinsic(r)
            return IntrinsicOutput(r_int=float(r.view(-1)[0].item()))

    def compute_batch(
        self, obs: Any, next_obs: Any | None = None, reduction: str = "none"
    ) -> Tensor:
        """Compute intrinsic for a batch of observations.

        If `next_obs` is provided, it is preferred (common RND practice),
        otherwise `obs` is used.
        """
        with torch.no_grad():
            x_src = next_obs if next_obs is not None else obs
            x = _as_tensor(x_src, self.device)
            r = self._intrinsic_raw_per_sample(x)
            r = self._normalize_intrinsic(r)
            if reduction == "mean":
                return r.mean()
            return r

    # ----------------------------- Training ----------------------------

    def loss(self, obs: Any) -> Mapping[str, Tensor]:
        """Compute predictor-vs-target MSE loss (no optimizer step)."""
        o = _as_tensor(obs, self.device)
        p, t = self._pred_and_targ(o)
        per = F.mse_loss(p, t, reduction="none").mean(dim=-1)  # [B]
        total = per.mean()

        # Update running RMS (EMA of r^2) for normalization diagnostics (no grad)
        with torch.no_grad():
            r2_batch_mean = (per**2).mean()
            self._r2_ema.mul_(self.cfg.rms_beta).add_((1.0 - self.cfg.rms_beta) * r2_batch_mean)

        return {"total": total, "intrinsic_mean": per.mean()}

    def update(self, obs: Any, steps: int = 1) -> Mapping[str, float]:
        """Optimize predictor for `steps` iterations on the same batch."""
        metrics: Mapping[str, float] = {}
        for _ in range(int(steps)):
            out = self.loss(obs)
            self._opt.zero_grad(set_to_none=True)
            out["total"].backward()
            nn.utils.clip_grad_norm_(
                self.predictor.parameters(), max_norm=float(self.cfg.grad_clip)
            )
            self._opt.step()

            metrics = {
                "loss_total": float(out["total"].detach().item()),
                "intrinsic_mean": float(out["intrinsic_mean"].detach().item()),
                "rms": float(torch.sqrt(self._r2_ema + self.cfg.rms_eps).detach().item()),
            }
        return metrics
