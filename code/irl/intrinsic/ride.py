"""RIDE intrinsic module (impact-only, embedding reused from ICM).

Sprint 2 — Step 1 implements the *impact* component of RIDE using the same
representation encoder φ(s) as ICM. The encoder and its training objective
are delegated to an internal ICM instance; we simply compute intrinsic reward
as the L2 distance in embedding space:

    r_impact = || φ(s_{t+1}) - φ(s_t) ||_2

Notes
-----
* Episodic binning / de-duplication (the denominator term in the full RIDE
  definition) is intentionally **not** implemented in this step; it will land
  in Sprint 2 — Step 2.
* The module exposes ICM’s optimization via `update(...)` so the shared encoder
  receives gradients from the inverse/forward objectives exactly as in ICM.
* Observation spaces are assumed to be vector Box spaces (image encoders will
  be added later, same as ICM).

API (aligned with ICM/RND for trainer integration)
--------------------------------------------------
- compute(tr) -> IntrinsicOutput
- compute_batch(obs, next_obs, actions=None, reduction="none") -> Tensor[[B] or [1]]
- loss(obs, next_obs, actions) -> dict(total, intrinsic_mean, icm_forward, icm_inverse)
- update(obs, next_obs, actions, steps=1) -> dict(loss_total, loss_forward, loss_inverse, intrinsic_mean)
"""

from __future__ import annotations

from typing import Any, Optional, Union

import gymnasium as gym
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import BaseIntrinsicModule, IntrinsicOutput, Transition
from .icm import ICM, ICMConfig


# ------------------------------ Small helpers ------------------------------


def _as_tensor(x: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype or torch.float32)


def _ensure_2d(x: Tensor) -> Tensor:
    """Ensure shape [B, D]; if [D], add batch dim; if [T,B,D], flatten to [T*B, D]."""
    if x.dim() == 1:
        return x.view(1, -1)
    if x.dim() == 2:
        return x
    return x.view(-1, x.size(-1))


# ---------------------------------- RIDE ------------------------------------


class RIDE(BaseIntrinsicModule, nn.Module):
    """RIDE (impact-only) using ICM encoder φ(s) and ICM training.

    Args:
        obs_space: gym.spaces.Box observation space (vector observations).
        act_space: Discrete or Box action space (required for ICM inverse head).
        device: torch device string or device.
        icm: Optionally supply a pre-built ICM instance to *share* its encoder.
             If None, an internal ICM is constructed with default ICMConfig.
        icm_cfg: Optional ICMConfig to configure the internal ICM if `icm` is None.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        device: Union[str, torch.device] = "cpu",
        icm: Optional[ICM] = None,
        icm_cfg: Optional[ICMConfig] = None,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("RIDE currently supports Box observation spaces (vector states).")
        self.device = torch.device(device)

        # Backing ICM module (owns encoder/training)
        self.icm = icm if icm is not None else ICM(obs_space, act_space, device=device, cfg=icm_cfg)

        # Convenience mirrors (no new parameters here)
        self.encoder = self.icm.encoder  # weight sharing
        self.is_discrete = self.icm.is_discrete
        self.obs_dim = int(obs_space.shape[0])

        self.to(self.device)

    # -------------------------- Intrinsic compute -------------------------

    @torch.no_grad()
    def _impact_per_sample(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        """Return per-sample impact = ||φ(s') - φ(s)||_2 with shape [B]."""
        o = _ensure_2d(obs)
        op = _ensure_2d(next_obs)
        phi_t = self.icm._phi(o)
        phi_tp1 = self.icm._phi(op)
        # L2 across embedding dims -> [B]
        return torch.norm(phi_tp1 - phi_t, p=2, dim=-1)

    def compute(self, tr: Transition) -> IntrinsicOutput:
        """Compute impact intrinsic for a single transition (no gradients)."""
        with torch.no_grad():
            s = _as_tensor(tr.s, self.device)
            sp = _as_tensor(tr.s_next, self.device)
            r = self._impact_per_sample(s.view(1, -1), sp.view(1, -1)).view(-1)[0]
            return IntrinsicOutput(r_int=float(r.item()))

    def compute_batch(
        self, obs: Any, next_obs: Any, actions: Any | None = None, reduction: str = "none"
    ) -> Tensor:
        """Vectorized impact intrinsic for batches.

        Returns:
            Tensor of shape [B] if reduction=="none"; scalar [1] if "mean".
        """
        with torch.no_grad():
            o = _as_tensor(obs, self.device)
            op = _as_tensor(next_obs, self.device)
            r = self._impact_per_sample(o, op)
            if reduction == "mean":
                return r.mean()
            return r

    # ----------------------------- Losses/Update ---------------------------

    def loss(self, obs: Any, next_obs: Any, actions: Any) -> dict[str, Tensor]:
        """Return ICM training loss + RIDE intrinsic mean (for diagnostics)."""
        # ICM losses train the shared encoder; no gradients from intrinsic itself
        icm_losses = self.icm.loss(obs, next_obs, actions)
        with torch.no_grad():
            o = _as_tensor(obs, self.device)
            op = _as_tensor(next_obs, self.device)
            r = self._impact_per_sample(o, op).mean()
        return {
            "total": icm_losses["total"],
            "icm_forward": icm_losses["forward"],
            "icm_inverse": icm_losses["inverse"],
            "intrinsic_mean": r,
        }

    def update(self, obs: Any, next_obs: Any, actions: Any, steps: int = 1) -> dict[str, float]:
        """Run ICM optimization steps; report ICM losses + RIDE intrinsic mean."""
        # Measure current intrinsic (for logging)
        with torch.no_grad():
            o = _as_tensor(obs, self.device)
            op = _as_tensor(next_obs, self.device)
            r_mean = self._impact_per_sample(o, op).mean().detach().item()

        # Delegate training to ICM (keeps encoder/inverse/forward in sync)
        metrics = self.icm.update(obs, next_obs, actions, steps=steps)
        out = {
            "loss_total": float(metrics["loss_total"]),
            "loss_forward": float(metrics["loss_forward"]),
            "loss_inverse": float(metrics["loss_inverse"]),
            "intrinsic_mean": float(r_mean),
        }
        return out
