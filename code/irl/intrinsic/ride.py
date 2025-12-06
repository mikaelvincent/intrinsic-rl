"""RIDE intrinsic (impact-only) with episodic binning.

Summary
-------
Intrinsic reward is the embedding change magnitude

    r = ||φ(s') − φ(s)||₂

where φ is the representation learned by a shared ICM backbone.

Optionally, per-episode de-duplication divides the raw impact by
``1 + N_ep(bin(φ(s')))`` to down-weight repeated visits to the same
embedding bin within an episode. The encoder / inverse / forward
heads are shared with ICM; only the ICM representation is trained
here.

Supports both vector and image observations: image inputs are routed
through the ICM's convolutional encoder path (handling HWC/CHW
layouts and scaling to [0, 1]).

Notes
-----
Historically this module exposed an ``obs_dim`` attribute derived from
``obs_space.shape[0]``. That is only meaningful for vector observations
and misleading for images (HWC/CHW). It has been removed in favour of
relying on the ICM encoder for shape handling.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import gymnasium as gym
import torch
from torch import Tensor, nn

from . import BaseIntrinsicModule, IntrinsicOutput, Transition
from .icm import ICM, ICMConfig
from irl.utils.torchops import as_tensor


class RIDE(BaseIntrinsicModule, nn.Module):
    """Impact-only intrinsic reward module.

    This module measures the change in the ICM embedding between
    consecutive states and (optionally) applies per-episode
    de-duplication via binning in embedding space.

    Parameters
    ----------
    obs_space :
        Observation space of the underlying environment (``gym.spaces.Box``).
    act_space :
        Action space used to construct the shared ICM backbone.
    device :
        Torch device on which the module's parameters are stored.
    icm :
        Optional pre-constructed :class:`ICM` instance. When omitted,
        a new ICM module is created internally.
    icm_cfg :
        Optional configuration used when instantiating an internal ICM.
        Ignored when ``icm`` is provided.
    bin_size :
        Width of each embedding bin used for episodic de-duplication.
        Larger bins group more states together.
    alpha_impact :
        Multiplicative scale applied to the raw impact signal.
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
            raise TypeError("RIDE supports Box observation spaces (vector or image).")
        self.device = torch.device(device)

        # Backing ICM module (owns encoder/training)
        self.icm = icm if icm is not None else ICM(obs_space, act_space, device=device, cfg=icm_cfg)

        # Convenience mirrors (no new parameters here)
        self.encoder = self.icm.encoder  # weight sharing
        self.is_discrete = self.icm.is_discrete

        # Episodic binning config
        self.bin_size: float = float(bin_size)
        self.alpha_impact: float = float(alpha_impact)

        # Per-env episodic counts; lazily sized on first call
        self._nvec: Optional[int] = None
        self._ep_counts: list[dict[Tuple[int, ...], int]] = []

        self.to(self.device)

    # -------------------------- Intrinsic compute -------------------------

    @torch.no_grad()
    def _impact_per_sample(self, obs: Any, next_obs: Any) -> Tensor:
        """Return per-sample impact ``||φ(s') − φ(s)||₂`` with shape ``[B]``.

        Accepts vector arrays or images (CHW/NCHW/NHWC); layout and
        scaling are handled by the underlying ICM encoder.
        """
        phi_t = self.icm._phi(obs)
        phi_tp1 = self.icm._phi(next_obs)
        return torch.norm(phi_tp1 - phi_t, p=2, dim=-1)

    def _ensure_counts(self, batch_size: int) -> None:
        """Initialise or resize per-environment episodic count tables."""
        if self._nvec is None or self._nvec != int(batch_size):
            self._nvec = int(batch_size)
            self._ep_counts = [dict() for _ in range(self._nvec)]

    @torch.no_grad()
    def _bin_keys(self, phi: Tensor) -> list[Tuple[int, ...]]:
        """Return integer bin keys for φ using ``floor(φ / bin_size)``."""
        bins = torch.floor(phi / float(self.bin_size)).to(dtype=torch.int64, device="cpu")
        return [tuple(map(int, row.tolist())) for row in bins]

    def compute(self, tr: Transition) -> IntrinsicOutput:
        """Compute intrinsic reward for a single transition.

        This variant returns *raw* impact only (no episodic de-duplication).
        """
        with torch.no_grad():
            r_raw = self._impact_per_sample(tr.s, tr.s_next).view(-1)[0]
            r = self.alpha_impact * r_raw
            return IntrinsicOutput(r_int=float(r.item()))

    def compute_batch(
        self, obs: Any, next_obs: Any, actions: Any | None = None, reduction: str = "none"
    ) -> Tensor:
        """Batch impact without episodic de-duplication.

        Parameters
        ----------
        obs, next_obs :
            Batch of observations and next observations (any layout accepted
            by the ICM encoder).
        actions :
            Unused; accepted for interface symmetry with other modules.
        reduction :
            Either ``"none"`` (default, returns shape ``[B]``) or ``"mean"``
            (returns a scalar).

        Returns
        -------
        torch.Tensor
            Impact values scaled by ``alpha_impact``.
        """
        with torch.no_grad():
            r = self._impact_per_sample(obs, next_obs)
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
        """Impact with episodic de-duplication.

        The episode-specific count table is reset for any environment
        whose ``done`` flag is ``True`` on this step. Within an episode,
        visits to the same embedding bin are down-weighted via

        .. math::

            r_t = \\frac{\\alpha_\\text{impact} \\cdot ||φ(s') - φ(s)||_2}
                        {1 + N_\\text{ep}(\\text{bin}(φ(s')))}

        Parameters
        ----------
        obs, next_obs :
            Batch of observations and next observations.
        dones :
            Boolean terminal flags (shape ``[B]``) indicating which
            environments terminated at this step.
        reduction :
            Either ``"none"`` (default, returns shape ``[B]``) or
            ``"mean"`` (returns a scalar).

        Returns
        -------
        torch.Tensor
            Binned impact values after per-episode de-duplication.
        """
        o = obs
        op = next_obs
        # Normalize shapes to batch-first on the fly via φ (no explicit reshape needed)
        phi_t = self.icm._phi(o)
        phi_tp1 = self.icm._phi(op)
        B = int(phi_tp1.size(0))
        self._ensure_counts(B)

        # Reset counts for envs that just finished
        if dones is not None:
            d = as_tensor(dones, device=torch.device("cpu"), dtype=torch.float32).view(-1)
            for i in range(B):
                if bool(d[i].item()):
                    self._ep_counts[i].clear()

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
        """Return ICM losses and a diagnostic mean raw impact.

        The intrinsic reward itself is not used in the loss; only the
        ICM encoder, inverse, and forward heads are trained.
        """
        icm_losses = self.icm.loss(obs, next_obs, actions)
        with torch.no_grad():
            r = self._impact_per_sample(obs, next_obs).mean()
        return {
            "total": icm_losses["total"],
            "icm_forward": icm_losses["forward"],
            "icm_inverse": icm_losses["inverse"],
            "intrinsic_mean": r,
        }

    def update(self, obs: Any, next_obs: Any, actions: Any, steps: int = 1) -> dict[str, float]:
        """Optimise the shared ICM on a fixed batch.

        Parameters
        ----------
        obs, next_obs, actions :
            Batch of transitions.
        steps :
            Number of optimisation passes over the same batch.

        Returns
        -------
        dict
            Scalar metrics containing ICM losses and the mean raw
            impact before scaling.
        """
        with torch.no_grad():
            r_mean = self._impact_per_sample(obs, next_obs).mean().detach().item()

        metrics = self.icm.update(obs, next_obs, actions, steps=steps)
        return {
            "loss_total": float(metrics["loss_total"]),
            "loss_forward": float(metrics["loss_forward"]),
            "loss_inverse": float(metrics["loss_inverse"]),
            "intrinsic_mean": float(r_mean),
        }
