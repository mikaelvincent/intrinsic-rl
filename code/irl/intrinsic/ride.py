"""RIDE intrinsic module (impact-only, embedding reused from ICM).

Sprint 2 — Step 1 implements the *impact* component of RIDE using the same
representation encoder φ(s) as ICM. The encoder and its training objective
are delegated to an internal ICM instance; we simply compute intrinsic reward
as the L2 distance in embedding space:

    r_impact = || φ(s_{t+1}) - φ(s_t) ||_2

Sprint 2 — Step 2 adds **episodic binning & counts** (§5.5):
    key = floor(φ / bin_size)        # elementwise; tuple used as dict key
    r_impact_binned = r_impact / (1 + N_ep(key_of(φ(s_{t+1}))))
    # counts are **per episode**, per parallel env, and reset on done.

Notes
-----
* Episodic binning is applied only by `compute_impact_binned(...)`, which
  accepts a per-env `dones` mask to reset the counts before computing the
  denominator for φ(s_{t+1}). The more generic `compute_batch(...)` keeps
  returning the raw impact (no binning), preserving backward compatibility
  for callers that operate on flattened batches.
* Episodic binning requires access to per‑env boundaries (the `dones` mask).
  In vectorized rollouts, call `compute_impact_binned` **once per step** with
  shape [B, D] for observations (see trainer integration).
* Observation spaces are assumed to be vector Box spaces (image encoders will
  be added later, same as ICM).

API (aligned with ICM/RND for trainer integration)
--------------------------------------------------
- compute(tr) -> IntrinsicOutput
- compute_batch(obs, next_obs, actions=None, reduction="none") -> Tensor[[B] or [1]]
  (raw impact; no binning)
- compute_impact_binned(obs, next_obs, dones=None, reduction="none") -> Tensor[[B] or [1]]
  (impact with episodic counts; new in Sprint 2 — Step 2)
- loss(obs, next_obs, actions) -> dict(total, intrinsic_mean, icm_forward, icm_inverse)
- update(obs, next_obs, actions, steps=1) -> dict(loss_total, loss_forward, loss_inverse, intrinsic_mean)
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple, Union

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
        bin_size: Size of bins in embedding space for episodic counting (default 0.25).
        alpha_impact: Optional scalar multiplier for impact reward (default 1.0).
    """

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        device: Union[str, torch.device] = "cpu",
        icm: Optional[ICM] = None,
        icm_cfg: Optional[ICMConfig] = None,
        *,
        bin_size: float = 0.25,
        alpha_impact: float = 1.0,
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

        # Episodic binning config
        self.bin_size: float = float(bin_size)
        self.alpha_impact: float = float(alpha_impact)

        # Per-env episodic counts; lazily sized on first call to compute_impact_binned
        self._nvec: Optional[int] = None
        self._ep_counts: list[dict[Tuple[int, ...], int]] = []

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

    def _ensure_counts(self, batch_size: int) -> None:
        """Initialize/resize per-env episodic counts structures."""
        if self._nvec is None or self._nvec != int(batch_size):
            self._nvec = int(batch_size)
            self._ep_counts = [dict() for _ in range(self._nvec)]

    @torch.no_grad()
    def _bin_keys(self, phi: Tensor) -> list[Tuple[int, ...]]:
        """Compute integer bin keys for each row in φ using floor(φ/bin_size)."""
        # [B, D] float32 -> [B, D] int via floor
        bins = torch.floor(phi / float(self.bin_size)).to(dtype=torch.int64, device="cpu")
        # Convert each row to a Python tuple[int,...] key
        return [tuple(map(int, row.tolist())) for row in bins]

    def compute(self, tr: Transition) -> IntrinsicOutput:
        """Compute impact intrinsic for a single transition (no gradients).

        Note: This single-sample API returns **alpha * raw impact** (no episodic
        binning, as episode boundaries are unknown here).
        """
        with torch.no_grad():
            s = _as_tensor(tr.s, self.device)
            sp = _as_tensor(tr.s_next, self.device)
            r_raw = self._impact_per_sample(s.view(1, -1), sp.view(1, -1)).view(-1)[0]
            r = self.alpha_impact * r_raw
            return IntrinsicOutput(r_int=float(r.item()))

    def compute_batch(
        self, obs: Any, next_obs: Any, actions: Any | None = None, reduction: str = "none"
    ) -> Tensor:
        """Vectorized impact intrinsic for batches (no episodic binning).

        Returns:
            Tensor of shape [B] if reduction=="none"; scalar [1] if "mean".
        """
        with torch.no_grad():
            o = _as_tensor(obs, self.device)
            op = _as_tensor(next_obs, self.device)
            r = self._impact_per_sample(o, op)
            r = self.alpha_impact * r
            if reduction == "mean":
                return r.mean()
            return r

    @torch.no_grad()
    def compute_impact_binned(
        self,
        obs: Any,
        next_obs: Any,
        dones: Any | None = None,
        reduction: str = "none",
    ) -> Tensor:
        """Impact intrinsic with **episodic binning & counts**.

        Args:
            obs: [B, obs_dim] observations at time t (normalized as trainer uses them).
            next_obs: [B, obs_dim] observations at time t+1 (normalized).
            dones: Optional boolean/0-1 array of shape [B]. For entries True,
                   the corresponding env's episodic counts are reset **before**
                   computing the denominator for φ(s_{t+1}) (new episode).
            reduction: "none" (default) -> [B], or "mean" -> scalar.

        Returns:
            Tensor of shape [B] if reduction=="none"; scalar if "mean".
        """
        o = _as_tensor(obs, self.device)
        op = _as_tensor(next_obs, self.device)
        o2 = _ensure_2d(o)
        op2 = _ensure_2d(op)
        B = int(op2.size(0))
        self._ensure_counts(B)

        # Reset counts for envs that just finished; Gym vector envs typically return s_{t+1}
        # from the next episode already, so we reset **before** using its bin key.
        if dones is not None:
            d = _as_tensor(dones, device=torch.device("cpu"), dtype=torch.float32).view(-1)
            for i in range(B):
                if bool(d[i].item()):
                    self._ep_counts[i].clear()

        # Compute embeddings and bin keys for next_obs (φ(s_{t+1}))
        phi_t = self.icm._phi(o2)
        phi_tp1 = self.icm._phi(op2)
        keys = self._bin_keys(phi_tp1)

        # Raw impact magnitude per sample
        raw = torch.norm(phi_tp1 - phi_t, p=2, dim=-1).to(device=self.device)

        # Apply episodic denominator 1 + N_ep(key) per env, then increment counts
        out = torch.empty_like(raw)
        for i in range(B):
            cnt = self._ep_counts[i].get(keys[i], 0)
            denom = 1.0 + float(cnt)
            out[i] = (self.alpha_impact * raw[i]) / denom
            # Increment visitation count for the bin of φ(s_{t+1})
            self._ep_counts[i][keys[i]] = cnt + 1

        if reduction == "mean":
            return out.mean()
        return out

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
