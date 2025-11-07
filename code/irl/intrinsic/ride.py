"""RIDE intrinsic (impact-only) with episodic binning.

Summary
-------
Intrinsic reward is the embedding change magnitude:

    r = ||φ(s') - φ(s)||_2

Optionally, per-episode de-duplication divides by (1 + N_ep(bin(φ(s')))).
The encoder/inverse/forward heads are shared with ICM; only the ICM
representation is trained here.

See: devspec/dev_spec_and_plan.md §5.3.4 (RIDE) and §5.5 (binning).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple, Union

import gymnasium as gym
import torch
from torch import Tensor, nn

from . import BaseIntrinsicModule, IntrinsicOutput, Transition
from .icm import ICM, ICMConfig
from irl.utils.torchops import as_tensor, ensure_2d


class RIDE(BaseIntrinsicModule, nn.Module):
    """Impact-only intrinsic using ICM encoder φ(s)."""

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
            raise TypeError("RIDE supports Box observation spaces (vector states).")
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

        # Per-env episodic counts; lazily sized on first call
        self._nvec: Optional[int] = None
        self._ep_counts: list[dict[Tuple[int, ...], int]] = []

        self.to(self.device)

    # -------------------------- Intrinsic compute -------------------------

    @torch.no_grad()
    def _impact_per_sample(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        """Return per-sample impact = ||φ(s') - φ(s)||₂ with shape [B]."""
        o = ensure_2d(obs)
        op = ensure_2d(next_obs)
        phi_t = self.icm._phi(o)
        phi_tp1 = self.icm._phi(op)
        return torch.norm(phi_tp1 - phi_t, p=2, dim=-1)

    def _ensure_counts(self, batch_size: int) -> None:
        """Initialize/resize per-env episodic counts structures."""
        if self._nvec is None or self._nvec != int(batch_size):
            self._nvec = int(batch_size)
            self._ep_counts = [dict() for _ in range(self._nvec)]

    @torch.no_grad()
    def _bin_keys(self, phi: Tensor) -> list[Tuple[int, ...]]:
        """Integer bin keys for φ using floor(φ / bin_size)."""
        bins = torch.floor(phi / float(self.bin_size)).to(dtype=torch.int64, device="cpu")
        return [tuple(map(int, row.tolist())) for row in bins]

    def compute(self, tr: Transition) -> IntrinsicOutput:
        """Single-transition intrinsic (raw impact; no episodic binning)."""
        with torch.no_grad():
            s = as_tensor(tr.s, self.device)
            sp = as_tensor(tr.s_next, self.device)
            r_raw = self._impact_per_sample(s.view(1, -1), sp.view(1, -1)).view(-1)[0]
            r = self.alpha_impact * r_raw
            return IntrinsicOutput(r_int=float(r.item()))

    def compute_batch(
        self, obs: Any, next_obs: Any, actions: Any | None = None, reduction: str = "none"
    ) -> Tensor:
        """Vectorized raw impact (no episodic binning)."""
        with torch.no_grad():
            o = as_tensor(obs, self.device)
            op = as_tensor(next_obs, self.device)
            r = self._impact_per_sample(o, op)
            r = self.alpha_impact * r
            return r.mean() if reduction == "mean" else r

    @torch.no_grad()
    def compute_impact_binned(
        self,
        obs: Any,
        next_obs: Any,
        dones: Any | None = None,
        reduction: str = "none",
    ) -> Tensor:
        """Impact with episodic de-duplication (counts reset on done=True)."""
        o = as_tensor(obs, self.device)
        op = as_tensor(next_obs, self.device)
        o2 = ensure_2d(o)
        op2 = ensure_2d(op)
        B = int(op2.size(0))
        self._ensure_counts(B)

        # Reset counts for envs that just finished
        if dones is not None:
            d = as_tensor(dones, device=torch.device("cpu"), dtype=torch.float32).view(-1)
            for i in range(B):
                if bool(d[i].item()):
                    self._ep_counts[i].clear()

        # Embeddings & keys for next_obs
        phi_t = self.icm._phi(o2)
        phi_tp1 = self.icm._phi(op2)
        keys = self._bin_keys(phi_tp1)

        # Raw impact
        raw = torch.norm(phi_tp1 - phi_t, p=2, dim=-1).to(device=self.device)

        # Apply denominator 1 + N_ep(key) and increment counts
        out = torch.empty_like(raw)
        for i in range(B):
            cnt = self._ep_counts[i].get(keys[i], 0)
            denom = 1.0 + float(cnt)
            out[i] = (self.alpha_impact * raw[i]) / denom
            self._ep_counts[i][keys[i]] = cnt + 1

        return out.mean() if reduction == "mean" else out

    # ----------------------------- Losses/Update ---------------------------

    def loss(self, obs: Any, next_obs: Any, actions: Any) -> dict[str, Tensor]:
        """ICM loss + mean raw impact (diagnostic)."""
        icm_losses = self.icm.loss(obs, next_obs, actions)
        with torch.no_grad():
            o = as_tensor(obs, self.device)
            op = as_tensor(next_obs, self.device)
            r = self._impact_per_sample(o, op).mean()
        return {
            "total": icm_losses["total"],
            "icm_forward": icm_losses["forward"],
            "icm_inverse": icm_losses["inverse"],
            "intrinsic_mean": r,
        }

    def update(self, obs: Any, next_obs: Any, actions: Any, steps: int = 1) -> dict[str, float]:
        """Optimize ICM; report ICM losses + current mean raw impact."""
        with torch.no_grad():
            o = as_tensor(obs, self.device)
            op = as_tensor(next_obs, self.device)
            r_mean = self._impact_per_sample(o, op).mean().detach().item()

        metrics = self.icm.update(obs, next_obs, actions, steps=steps)
        return {
            "loss_total": float(metrics["loss_total"]),
            "loss_forward": float(metrics["loss_forward"]),
            "loss_inverse": float(metrics["loss_inverse"]),
            "intrinsic_mean": float(r_mean),
        }
