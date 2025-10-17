"""Proposed unified intrinsic module (Step 1: RIDE + R‑IAC combination).

This module combines:
  • RIDE‑style *impact*:   ‖φ(s') − φ(s)‖₂
  • R‑IAC *learning progress* per region i: LP_i = max(0, EMA_long_i − EMA_short_i)

Returned (unnormalized) intrinsic per transition:
    r_int = α_impact * r_impact + α_LP * LP_i

Notes
-----
* This file implements **only Step 1** of Sprint 4 (no gating/hysteresis yet).
* The module **does not** normalize outputs internally; the trainer's global
  RMS normalizer will scale `r_int` (see irl.train). Later steps may introduce
  per-component normalization and gating.
* An internal ICM instance provides the shared representation φ(s), the inverse
  head (for training) and the forward head used to compute prediction error.

Public API (aligned with other intrinsic modules)
-------------------------------------------------
- compute(tr) -> IntrinsicOutput
- compute_batch(obs, next_obs, actions, reduction="none") -> Tensor[[B] or scalar]
- loss(obs, next_obs, actions) -> dict(total, icm_forward, icm_inverse, intrinsic_mean)
- update(obs, next_obs, actions, steps=1) -> dict(loss_total, loss_forward, loss_inverse, intrinsic_mean)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, Iterable

import numpy as np
import gymnasium as gym
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import BaseIntrinsicModule, IntrinsicOutput, Transition
from .icm import ICM, ICMConfig
from .regions import KDTreeRegionStore


# ------------------------------ Small helpers ------------------------------


def _as_tensor(x: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype or torch.float32)


def _ensure_2d(x: Tensor) -> Tensor:
    """Ensure [B, D]; if [D], add batch dim; if [T,B,D], flatten to [T*B, D]."""
    if x.dim() == 1:
        return x.view(1, -1)
    if x.dim() == 2:
        return x
    return x.view(-1, x.size(-1))


@dataclass
class _RegionStats:
    """Per-region EMA container (unnormalized for Step 1)."""

    ema_long: float = 0.0
    ema_short: float = 0.0
    count: int = 0


# -------------------------------- Proposed ---------------------------------


class Proposed(BaseIntrinsicModule, nn.Module):
    """Unified intrinsic = α_impact·impact + α_LP·LP(region).

    Args:
        obs_space: Box observation space (vector states).
        act_space: Discrete or Box action space (for ICM inverse/forward).
        device: torch device string or device.
        icm: Optional external ICM to share encoder/heads; if None, create one.
        icm_cfg: Optional ICMConfig used when constructing an internal ICM.
        alpha_impact: Multiplier for RIDE‑style impact term.
        alpha_lp: Multiplier for region LP term.
        region_capacity: KD-tree leaf capacity before split.
        depth_max: Maximum KD-tree depth.
        ema_beta_long / ema_beta_short: EMA coefficients for LP.

    Outputs are **not normalized** here; trainer will apply global RMS.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        device: Union[str, torch.device] = "cpu",
        icm: Optional[ICM] = None,
        icm_cfg: Optional[ICMConfig] = None,
        *,
        alpha_impact: float = 1.0,
        alpha_lp: float = 0.5,
        region_capacity: int = 200,
        depth_max: int = 12,
        ema_beta_long: float = 0.995,
        ema_beta_short: float = 0.90,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("Proposed currently supports Box observation spaces (vector states).")
        self.device = torch.device(device)

        # Backing ICM (shared representation φ, forward/inverse heads)
        self.icm = icm if icm is not None else ICM(obs_space, act_space, device=device, cfg=icm_cfg)
        self.encoder = self.icm.encoder  # explicit alias
        self.is_discrete = self.icm.is_discrete
        self.obs_dim = int(obs_space.shape[0])
        self.phi_dim = int(self.icm.cfg.phi_dim)

        # R‑IAC regionization in φ-space
        self.store = KDTreeRegionStore(
            dim=self.phi_dim, capacity=int(region_capacity), depth_max=int(depth_max)
        )
        self._stats: dict[int, _RegionStats] = {}

        # Coefficients / EMA knobs
        self.alpha_impact = float(alpha_impact)
        self.alpha_lp = float(alpha_lp)
        self.beta_long = float(ema_beta_long)
        self.beta_short = float(ema_beta_short)

        # Outputs are NOT pre-normalized in Step 1
        self.outputs_normalized: bool = False

        self.to(self.device)

    # -------------------------- Core components --------------------------

    @torch.no_grad()
    def _impact_per_sample(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        """RIDE impact magnitude: ||φ(s') - φ(s)||₂, shape [B]."""
        o = _ensure_2d(obs)
        op = _ensure_2d(next_obs)
        phi_t = self.icm._phi(o)
        phi_tp1 = self.icm._phi(op)
        return torch.norm(phi_tp1 - phi_t, p=2, dim=-1)

    @torch.no_grad()
    def _forward_error_and_phi(self, obs: Tensor, next_obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (per-sample forward MSE in φ space [B], phi(s) [B,D])."""
        o = _ensure_2d(obs)
        op = _ensure_2d(next_obs)
        a = actions
        phi_t = self.icm._phi(o)
        phi_tp1 = self.icm._phi(op)
        a_fwd = self.icm._act_for_forward(a)
        pred = self.icm.forward_head(torch.cat([phi_t, a_fwd], dim=-1))
        per_dim = F.mse_loss(pred, phi_tp1, reduction="none")  # [B,D]
        return per_dim.mean(dim=-1), phi_t  # [B], [B,D]

    def _update_region(self, rid: int, error: float) -> float:
        """Update EMAs for region rid with forward error; return LP (>=0)."""
        st = self._stats.get(rid)
        if st is None or st.count == 0:
            self._stats[rid] = _RegionStats(ema_long=error, ema_short=error, count=1)
            return 0.0  # first observation: define LP=0
        st.ema_long = self.beta_long * st.ema_long + (1.0 - self.beta_long) * error
        st.ema_short = self.beta_short * st.ema_short + (1.0 - self.beta_short) * error
        st.count += 1
        return max(0.0, st.ema_long - st.ema_short)

    # -------------------------- Intrinsic compute -------------------------

    def compute(self, tr: Transition) -> IntrinsicOutput:
        """Compute α_imp·impact + α_LP·LP(region(φ(s))) for a single transition."""
        with torch.no_grad():
            s = _as_tensor(tr.s, self.device)
            sp = _as_tensor(tr.s_next, self.device)
            a = _as_tensor(tr.a, self.device)

            # Components
            impact = self._impact_per_sample(s.view(1, -1), sp.view(1, -1)).view(-1)[0]
            err, phi_t = self._forward_error_and_phi(
                s.view(1, -1), sp.view(1, -1), a.view(1, -1 if not self.is_discrete else 1)
            )
            rid = int(self.store.insert(phi_t.detach().cpu().numpy().reshape(-1)))
            lp = self._update_region(rid, float(err.item()))

            r = self.alpha_impact * float(impact.item()) + self.alpha_lp * float(lp)
            return IntrinsicOutput(r_int=float(r))

    @torch.no_grad()
    def compute_batch(
        self,
        obs: Any,
        next_obs: Any,
        actions: Any,
        reduction: str = "none",
    ) -> Tensor:
        """Vectorized intrinsic for a batch (updates RIAC EMAs internally)."""
        o = _as_tensor(obs, self.device)
        op = _as_tensor(next_obs, self.device)
        a = _as_tensor(actions, self.device)

        # Components
        impact = self._impact_per_sample(o, op)  # [B]
        err, phi_t = self._forward_error_and_phi(o, op, a)  # [B], [B,D]

        # Region IDs and LP updates
        phi_np = phi_t.detach().cpu().numpy()
        rids = self.store.bulk_insert(phi_np)  # [B]
        lp_vals: list[float] = []
        for i in range(impact.shape[0]):
            lp = self._update_region(int(rids[i]), float(err[i].item()))
            lp_vals.append(lp)
        lp_t = torch.as_tensor(lp_vals, dtype=torch.float32, device=self.device)  # [B]

        out = self.alpha_impact * impact + self.alpha_lp * lp_t  # [B]
        if reduction == "mean":
            return out.mean()
        return out

    # ----------------------------- Loss & Update ---------------------------

    def loss(self, obs: Any, next_obs: Any, actions: Any) -> dict[str, Tensor]:
        """ICM training loss + diagnostic r_int mean (non-mutating)."""
        icm_losses = self.icm.loss(obs, next_obs, actions)

        with torch.no_grad():
            o = _as_tensor(obs, self.device)
            op = _as_tensor(next_obs, self.device)
            a = _as_tensor(actions, self.device)

            impact = self._impact_per_sample(o, op)  # [B]
            err, phi_t = self._forward_error_and_phi(o, op, a)  # [B], [B,D]

            # Estimate LP without mutating EMAs: use current region stats
            rids = [self.store.locate(p) for p in phi_t.detach().cpu().numpy()]
            lp_now = []
            for rid in rids:
                st = self._stats.get(int(rid))
                raw_lp = (
                    0.0 if st is None or st.count == 0 else max(0.0, st.ema_long - st.ema_short)
                )
                lp_now.append(raw_lp)

            r_mean = (self.alpha_impact * impact + self.alpha_lp * torch.as_tensor(
                lp_now, dtype=torch.float32, device=self.device
            )).mean()

        return {
            "total": icm_losses["total"],
            "icm_forward": icm_losses["forward"],
            "icm_inverse": icm_losses["inverse"],
            "intrinsic_mean": r_mean,
        }

    def update(self, obs: Any, next_obs: Any, actions: Any, steps: int = 1) -> dict[str, float]:
        """Train ICM (shared encoder/heads) and report losses + r_int mean (non-mutating)."""
        # Diagnostic r_int mean before update (non-mutating LP snapshot)
        with torch.no_grad():
            o = _as_tensor(obs, self.device)
            op = _as_tensor(next_obs, self.device)
            a = _as_tensor(actions, self.device)

            impact = self._impact_per_sample(o, op)  # [B]
            err, phi_t = self._forward_error_and_phi(o, op, a)
            rids = [self.store.locate(p) for p in phi_t.detach().cpu().numpy()]
            lp_now = []
            for rid in rids:
                st = self._stats.get(int(rid))
                raw_lp = (
                    0.0 if st is None or st.count == 0 else max(0.0, st.ema_long - st.ema_short)
                )
                lp_now.append(raw_lp)
            r_mean = float((self.alpha_impact * impact + self.alpha_lp * torch.as_tensor(
                lp_now, dtype=torch.float32, device=self.device
            )).mean().item())

        metrics = self.icm.update(obs, next_obs, actions, steps=steps)
        return {
            "loss_total": float(metrics["loss_total"]),
            "loss_forward": float(metrics["loss_forward"]),
            "loss_inverse": float(metrics["loss_inverse"]),
            "intrinsic_mean": float(r_mean),
        }
