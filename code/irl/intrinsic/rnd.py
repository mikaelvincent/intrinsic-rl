"""Random Network Distillation (RND).

Intrinsic is the MSE between a trainable predictor and a fixed random target
on observations (prefers next_obs if provided). Optional running RMS can
normalize outputs locally; by default the trainer handles global scaling.

This module participates in the unified normalization contract:
- `self.outputs_normalized` reflects whether intrinsic values are already
  normalized *inside* the module. When True, the trainer must NOT apply
  global normalization again (only clip+scale).
- For diagnostics, `.rms` exposes the current RMS used when internal
  normalization is enabled.

See: devspec/dev_spec_and_plan.md §5.3.3.
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
from irl.utils.torchops import as_tensor, ensure_2d


@dataclass
class RNDConfig:
    """Lightweight configuration for RND."""

    feature_dim: int = 128
    hidden: Tuple[int, int] = (256, 256)
    lr: float = 3e-4
    grad_clip: float = 5.0
    rms_beta: float = 0.99
    rms_eps: float = 1e-8
    # When True, the module normalizes intrinsic internally and sets
    # `outputs_normalized=True` so the trainer will not normalize again.
    normalize_intrinsic: bool = False


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
            raise TypeError("RND supports Box observation spaces (vector states).")

        self.device = torch.device(device)
        self.cfg = cfg or RNDConfig()
        self.obs_dim = int(obs_space.shape[0])

        # Target: fixed, randomly initialized; Predictor: trainable
        self.target = mlp(self.obs_dim, self.cfg.hidden, out_dim=self.cfg.feature_dim)
        self.predictor = mlp(self.obs_dim, self.cfg.hidden, out_dim=self.cfg.feature_dim)

        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()

        self._opt = torch.optim.Adam(self.predictor.parameters(), lr=float(self.cfg.lr))

        # Running RMS over unnormalized intrinsic values
        self.register_buffer("_r2_ema", torch.tensor(1.0, dtype=torch.float32))  # start non-zero

        # ---- Unified normalization contract --------------------------------
        # If normalize_intrinsic=True, the module's outputs are already normalized.
        # The trainer should then skip its own global RMS normalization.
        self.outputs_normalized: bool = bool(self.cfg.normalize_intrinsic)

        self.to(self.device)

    # -------------------------- Diagnostics ---------------------------

    @property
    def rms(self) -> float:
        """Current RMS used by the internal normalizer (if enabled)."""
        return float(torch.sqrt(self._r2_ema + self.cfg.rms_eps).detach().item())

    # -------------------------- Core compute ---------------------------

    def _pred_and_targ(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x2 = ensure_2d(x)
        p = self.predictor(x2)
        with torch.no_grad():
            t = self.target(x2)
        return p, t

    def _intrinsic_raw_per_sample(self, x: Tensor) -> Tensor:
        """Per-sample MSE, shape [B]."""
        p, t = self._pred_and_targ(x)
        return F.mse_loss(p, t, reduction="none").mean(dim=-1)

    def _normalize_intrinsic(self, r: Tensor) -> Tensor:
        """Optionally normalize with the module's running RMS."""
        if not self.cfg.normalize_intrinsic:
            return r
        denom = torch.sqrt(self._r2_ema + self.cfg.rms_eps)
        return r / denom

    # Public API

    def compute(self, tr) -> IntrinsicOutput:
        """Compute intrinsic for a single transition (prefers s')."""
        with torch.no_grad():
            x = tr.s_next if hasattr(tr, "s_next") and tr.s_next is not None else tr.s
            xt = as_tensor(x, self.device)
            r = self._intrinsic_raw_per_sample(xt)
            r = self._normalize_intrinsic(r)
            return IntrinsicOutput(r_int=float(r.view(-1)[0].item()))

    def compute_batch(
        self, obs: Any, next_obs: Any | None = None, reduction: str = "none"
    ) -> Tensor:
        """Batch intrinsic; if next_obs is provided, it is preferred."""
        with torch.no_grad():
            x_src = next_obs if next_obs is not None else obs
            x = as_tensor(x_src, self.device)
            r = self._intrinsic_raw_per_sample(x)
            r = self._normalize_intrinsic(r)
            return r.mean() if reduction == "mean" else r

    # ----------------------------- Training ----------------------------

    def loss(self, obs: Any) -> Mapping[str, Tensor]:
        """Predictor-vs-target MSE and running RMS diagnostic."""
        o = as_tensor(obs, self.device)
        p, t = self._pred_and_targ(o)
        per = F.mse_loss(p, t, reduction="none").mean(dim=-1)  # [B]
        total = per.mean()

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
                "rms": self.rms,
            }
        return metrics
