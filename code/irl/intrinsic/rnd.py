"""
Random Network Distillation (RND).

Intrinsic reward is the mean-squared error between a trainable
predictor network and a fixed, randomly initialised target network
evaluated on observations (preferring ``next_obs`` when provided).

A lightweight running RMS normaliser can be enabled inside this
module; otherwise the trainer applies a global normalisation layer.

Normalization contract
----------------------
This module participates in a unified normalisation contract:

* ``self.outputs_normalized`` indicates whether intrinsic values are
  already normalised inside the module. When ``True``, the trainer
  must not normalise them again (only clip and scale).
* For diagnostics, :attr:`RND.rms` exposes the current RMS used when
  internal normalisation is enabled.
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
from irl.models.cnn import ConvEncoder, ConvEncoderConfig
from irl.utils.torchops import as_tensor, ensure_2d
from irl.utils.image import preprocess_image, ImagePreprocessConfig


@dataclass
class RNDConfig:
    """Configuration for :class:`RND`.

    Parameters
    ----------
    feature_dim :
        Size of the latent feature vector produced by the target
        and predictor networks.
    hidden :
        Hidden-layer sizes for the shared MLP backbone (vector obs only).
    lr :
        Learning rate for the Adam optimiser.
    grad_clip :
        Maximum gradient norm for predictor updates.
    rms_beta :
        Decay for the running RMS when internal normalisation is used.
    rms_eps :
        Small epsilon added under the square root for numerical
        stability in the RMS.
    normalize_intrinsic :
        When ``True``, the module normalises intrinsic values
        internally and exposes ``outputs_normalized=True`` so the
        trainer can skip its global normaliser.
    """

    feature_dim: int = 128
    hidden: Tuple[int, int] = (256, 256)  # vector path only
    lr: float = 3e-4
    grad_clip: float = 5.0
    rms_beta: float = 0.99
    rms_eps: float = 1e-8
    # When True, the module normalizes intrinsic internally and sets
    # `outputs_normalized=True` so the trainer will not normalize again.
    normalize_intrinsic: bool = False


class RND(BaseIntrinsicModule, nn.Module):
    """Random Network Distillation intrinsic module.

    Parameters
    ----------
    obs_space :
        Observation space (``gym.spaces.Box``). Supports both vector and image
        observations. Image inputs are routed through a :class:`ConvEncoder`
        with centralised preprocessing; vector inputs use an MLP backbone.
    device :
        Torch device on which parameters and buffers are stored.
    cfg :
        Optional :class:`RNDConfig` instance. If ``None``, a default
        configuration is used.

    Notes
    -----
    The target network is fixed after initialisation. Only the
    predictor network is trained to minimise the prediction error,
    which becomes the intrinsic reward signal.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        device: Union[str, torch.device] = "cpu",
        cfg: Optional[RNDConfig] = None,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("RND supports Box observation spaces (vector or images).")

        self.device = torch.device(device)
        self.cfg = cfg or RNDConfig()

        # Determine observation type
        self.is_image = len(obs_space.shape) >= 2

        if self.is_image:
            # Infer channel count from either leading or trailing axis (CHW vs HWC)
            shape = tuple(int(s) for s in obs_space.shape)
            if len(shape) == 3:
                c0 = shape[0]
                c2 = shape[-1]
                if c0 in (1, 3, 4) and c2 not in (1, 3, 4):
                    in_channels, in_hw = c0, (shape[1], shape[2])
                else:
                    in_channels, in_hw = c2, (shape[0], shape[1])
            else:
                in_channels = shape[-1]
                in_hw = (shape[0], shape[1])

            # CNN target/predictor (same architecture; target frozen)
            cnn_cfg = ConvEncoderConfig(in_channels=int(in_channels), out_dim=int(self.cfg.feature_dim))
            # Pass in_hw so final projection is created immediately (checkpoint compat)
            self.target = ConvEncoder(cnn_cfg, in_hw=in_hw)
            self.predictor = ConvEncoder(cnn_cfg, in_hw=in_hw)

            for p in self.target.parameters():
                p.requires_grad = False
            self.target.eval()

            # Centralized preprocessing for images -> NCHW float32 in [0, 1]
            self._img_pre_cfg = ImagePreprocessConfig(
                grayscale=False,
                scale_uint8=True,
                normalize_mean=None,
                normalize_std=None,
                channels_first=True,
            )
        else:
            # Vector path: simple MLPs to feature_dim
            self.obs_dim = int(obs_space.shape[0])
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
        """Current RMS used by the internal normaliser (if enabled)."""
        return float(torch.sqrt(self._r2_ema + self.cfg.rms_eps).detach().item())

    # -------------------------- Core compute ---------------------------

    def _pred_and_targ(self, x: Tensor | object) -> Tuple[Tensor, Tensor]:
        """Return predictor and target features for a batch of inputs."""
        if self.is_image:
            # Accept HWC/NHWC/CHW/NCHW with optional leading dims; collapse to NCHW
            t = x if torch.is_tensor(x) else torch.as_tensor(x)
            if t.dim() >= 5:
                t = t.reshape(-1, *t.shape[-3:])
            img = preprocess_image(
                t, cfg=self._img_pre_cfg, device=self.device
            )  # -> NCHW float32 [0,1]
            p = self.predictor(img)
            with torch.no_grad():
                t = self.target(img)
            return p, t
        else:
            x2 = ensure_2d(as_tensor(x, self.device))
            p = self.predictor(x2)
            with torch.no_grad():
                t = self.target(x2)
            return p, t

    def _intrinsic_raw_per_sample(self, x: Tensor | object) -> Tensor:
        """Per-sample mean-squared error between predictor and target.

        Returns a tensor of shape ``[B]``.
        """
        p, t = self._pred_and_targ(x)
        return F.mse_loss(p, t, reduction="none").mean(dim=-1)

    def _update_rms_from_raw(self, per: Tensor) -> None:
        """Update the internal RMS EMA from a batch of unnormalised values.

        This helper centralises RMS updates so that both training
        (``loss``/``update``) and intrinsic computation paths can opt in
        without duplicating the EMA logic.
        """
        if per.numel() == 0:
            return
        with torch.no_grad():
            vals = per.to(device=self._r2_ema.device, dtype=torch.float32)
            r2_batch_mean = (vals**2).mean()
            self._r2_ema.mul_(self.cfg.rms_beta).add_(
                (1.0 - self.cfg.rms_beta) * r2_batch_mean
            )

    def _normalize_intrinsic(self, r: Tensor) -> Tensor:
        """Optionally normalise intrinsic values with the module's RMS.

        When ``cfg.normalize_intrinsic`` is ``False``, the input tensor
        is returned unchanged.
        """
        if not self.cfg.normalize_intrinsic:
            return r
        denom = torch.sqrt(self._r2_ema + self.cfg.rms_eps)
        return r / denom

    # Public API

    def compute(self, tr) -> IntrinsicOutput:
        """Compute intrinsic reward for a single transition.

        Prefers ``tr.s_next`` when available, otherwise uses ``tr.s``.
        """
        with torch.no_grad():
            x = tr.s_next if hasattr(tr, "s_next") and tr.s_next is not None else tr.s
            r_raw = self._intrinsic_raw_per_sample(x)
            if self.cfg.normalize_intrinsic:
                self._update_rms_from_raw(r_raw)
            r = self._normalize_intrinsic(r_raw)
            return IntrinsicOutput(r_int=float(r.view(-1)[0].item()))

    def compute_batch(
        self, obs: Any, next_obs: Any | None = None, reduction: str = "none"
    ) -> Tensor:
        """Batch intrinsic computation.

        Parameters
        ----------
        obs :
            Batch of observations used when ``next_obs`` is ``None``.
        next_obs :
            Batch of next observations, preferred when provided.
        reduction :
            Either ``"none"`` (default, returns shape ``[B]``) or ``"mean"``
            (returns a scalar).

        Returns
        -------
        torch.Tensor
            Per-sample or mean intrinsic values after optional
            internal normalisation.
        """
        with torch.no_grad():
            x_src = next_obs if next_obs is not None else obs
            r_raw = self._intrinsic_raw_per_sample(x_src)
            if self.cfg.normalize_intrinsic:
                self._update_rms_from_raw(r_raw)
            r = self._normalize_intrinsic(r_raw)
            return r.mean() if reduction == "mean" else r

    # ----------------------------- Training ----------------------------

    def loss(self, obs: Any) -> Mapping[str, Tensor]:
        """Predictor–target MSE and running RMS diagnostic.

        Parameters
        ----------
        obs :
            Batch of observations.

        Returns
        -------
        Mapping[str, torch.Tensor]
            Mapping with keys ``"total"`` (scalar loss) and
            ``"intrinsic_mean"`` (mean unnormalised intrinsic).
        """
        p, t = self._pred_and_targ(obs)
        per = F.mse_loss(p, t, reduction="none").mean(dim=-1)  # [B]
        total = per.mean()

        # For the default (normalize_intrinsic=False) path, keep updating the
        # internal RMS here so `rms` remains a useful diagnostic while the
        # trainer applies its own global normaliser. When intrinsic
        # normalisation is enabled, RMS is maintained in compute/compute_batch
        # to avoid double-updating on the same batch.
        if not self.cfg.normalize_intrinsic:
            self._update_rms_from_raw(per)

        return {"total": total, "intrinsic_mean": per.mean()}

    def update(self, obs: Any, steps: int = 1) -> Mapping[str, float]:
        """Optimise the predictor network on a fixed batch.

        Parameters
        ----------
        obs :
            Batch of observations.
        steps :
            Number of optimisation passes over the same batch.

        Returns
        -------
        Mapping[str, float]
            Dictionary containing the final loss, mean intrinsic, and
            current RMS estimate.
        """
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
                "loss_intrinsic_mean": float(out["intrinsic_mean"].detach().item()),
                "rms": self.rms,
            }
        return metrics
