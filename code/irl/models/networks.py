"""MLP/CNN-based policy and value networks for PPO.

This module provides neural network backbones that support both vector and
image observations:

* Vector ``Box`` observations → MLP backbone.
* Image ``Box`` observations (rank ≥ 2) → :class:`ConvEncoder` backbone
  (tolerant of NCHW/NHWC layouts with automatic 0–1 scaling).

Discrete action spaces use a categorical policy distribution, while
continuous ``Box`` action spaces use a diagonal Gaussian with a
state-independent log standard deviation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Tuple, Union

import gymnasium as gym
import torch
from torch import Tensor, nn

from .distributions import CategoricalDist, DiagGaussianDist
from .layers import mlp  # re-exported for backward compatibility
from .cnn import ConvEncoder, ConvEncoderConfig
from irl.utils.image import preprocess_image, ImagePreprocessConfig


def _space_dims(space: gym.Space) -> int:
    """Return the dimensionality associated with a Gymnasium space.

    For ``Box`` spaces this is the size of the last dimension. For
    ``Discrete`` spaces this is the number of actions.
    """
    if isinstance(space, gym.spaces.Box):
        return int(space.shape[0])
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    raise TypeError(f"Unsupported action/observation space type: {type(space)}")


def _is_image_space(space: gym.Space) -> bool:
    """Return ``True`` if a ``Box`` space is treated as image-like."""
    return isinstance(space, gym.spaces.Box) and len(space.shape) >= 2


def _infer_c_hw(shape: tuple[int, ...]) -> Tuple[int, Tuple[int, int]]:
    """Infer (C, (H, W)) from a shape tuple (assuming 3D).

    Heuristic: if dim 0 is small (1, 3, 4), assume CHW. Otherwise HWC.
    """
    if len(shape) != 3:
        # Fallback for non-3D shapes: treat as square-ish or fail downstream
        # Usually Gym images are 3D.
        return shape[-1], (shape[0], shape[1]) if len(shape) >= 2 else (0, 0)

    c0 = shape[0]
    c2 = shape[-1]
    # If c0 is a typical channel count and c2 is large, assume CHW.
    if c0 in (1, 3, 4) and c2 not in (1, 3, 4):
        return c0, (shape[1], shape[2])
    # Default to HWC
    return c2, (shape[0], shape[1])


def _to_tensor(x: Tensor | object, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """Convert an input to a tensor on ``device`` with the requested ``dtype``."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _prep_vector(x: Tensor, obs_dim: int) -> Tensor:
    """Return a ``[N, obs_dim]`` view for vector observations.

    Inputs with extra leading dimensions are flattened along all but the
    final feature dimension.
    """
    t = x
    if t.dim() == 1:
        return t.view(1, obs_dim)
    if t.dim() == 2:
        return t
    # Flatten all leading dims except the last feature dim
    return t.view(-1, obs_dim)


def _prep_images_to_nchw(x: Tensor | object, expected_c: int, device: torch.device) -> Tensor:
    """Return ``[N, C, H, W]`` float32 images in ``[0, 1]`` from arbitrary layouts.

    This is a centralised pathway that delegates layout and dtype handling
    to :func:`irl.utils.image.preprocess_image`. It accepts inputs in HWC,
    CHW, NHWC, or NCHW format and collapses any extra leading dimensions,
    for example ``(T, B, H, W, C)``.
    """
    # Convert to a tensor only to inspect rank; avoid forcing float32 here so we retain uint8 scaling semantics.
    xt = x if torch.is_tensor(x) else torch.as_tensor(x)
    if xt.dim() >= 5:
        # Collapse leading dims to [N, H, W, C] or [N, C, H, W]
        xt = xt.reshape(-1, *xt.shape[-3:])
    # Use a shared, explicit preprocessing config: keep channel count as-is (no grayscale), channels-first output.
    cfg = ImagePreprocessConfig(
        grayscale=False,
        scale_uint8=True,
        normalize_mean=None,
        normalize_std=None,
        channels_first=True,
    )
    out = preprocess_image(xt, cfg=cfg, device=device)  # -> NCHW float32 in [0,1]
    # Sanity: channel count must match ConvEncoder's configured in_channels to avoid silent mismatches.
    c = int(out.shape[1])
    if c != int(expected_c):
        raise ValueError(
            f"Image channel mismatch: model expects C={expected_c}, but preprocessed input has C={c}. "
            "Ensure the observation space shape matches the configured encoder channels."
        )
    return out


# -------------------- Networks --------------------


class PolicyNetwork(nn.Module):
    """PPO policy with MLP (vector) or CNN (image) backbone.

    Parameters
    ----------
    obs_space :
        Observation space of the environment. Must be a ``Box``; spaces
        with rank ≥ 2 are treated as image observations.
    action_space :
        Action space for the policy. Discrete spaces produce a
        :class:`CategoricalDist`, while continuous ``Box`` spaces produce
        a :class:`DiagGaussianDist`.
    hidden_sizes :
        Hidden-layer sizes for the MLP backbone used with vector
        observations.
    cnn_cfg :
        Optional :class:`ConvEncoderConfig` used for image observations.
        When provided, its ``in_channels`` field is overridden to match
        the number of channels implied by ``obs_space``.
    """

    def __init__(
        self,
        self_obs_space: gym.Space,  # renamed param to avoid shadowing
        action_space: gym.Space,
        hidden_sizes: Iterable[int] = (256, 256),
        cnn_cfg: ConvEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        # Use arg name compatibility for users passing by keyword (obs_space) if needed,
        # though strictly we use positionals.
        obs_space = self_obs_space

        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("PolicyNetwork supports Box observations (vector or images).")

        self.is_image = _is_image_space(obs_space)
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)

        if self.is_image:
            in_ch, in_hw = _infer_c_hw(tuple(int(s) for s in obs_space.shape))
            # Honor user-supplied ConvEncoderConfig; override only in_channels to match the space.
            cfg = cnn_cfg if cnn_cfg is not None else ConvEncoderConfig(in_channels=in_ch)
            if int(cfg.in_channels) != int(in_ch):
                cfg = replace(cfg, in_channels=int(in_ch))
            # Pass in_hw so projection is initialized immediately
            self.cnn = ConvEncoder(cfg, in_hw=in_hw)
            feat_dim = int(self.cnn.out_dim)
        else:
            self.obs_dim = int(obs_space.shape[0])
            self.backbone = mlp(self.obs_dim, tuple(hidden_sizes), out_dim=None)
            feat_dim = int(list(hidden_sizes)[-1]) if isinstance(hidden_sizes, (list, tuple)) else 256

        if self.is_discrete:
            self.n_actions = int(_space_dims(action_space))
            self.policy_head = nn.Linear(feat_dim, self.n_actions)
        else:
            if not isinstance(action_space, gym.spaces.Box):
                raise TypeError(f"Unsupported action space: {type(action_space)}")
            self.act_dim = int(_space_dims(action_space))
            self.mu_head = nn.Linear(feat_dim, self.act_dim)
            self.log_std = nn.Parameter(torch.zeros(self.act_dim))

    # ----- API -----

    def distribution(self, obs: Tensor | object) -> Union[CategoricalDist, DiagGaussianDist]:
        """Return the action distribution induced by the current policy.

        Parameters
        ----------
        obs :
            Batch of observations. For image observations this can be in
            HWC, NHWC, CHW, or NCHW format with optional leading time or
            batch dimensions.

        Returns
        -------
        CategoricalDist or DiagGaussianDist
            A distribution object that provides ``sample()``,
            ``log_prob()``, and ``entropy()``.
        """
        device = self.log_std.device if hasattr(self, "log_std") else next(self.parameters()).device  # type: ignore[union-attr]
        if self.is_image:
            x = _prep_images_to_nchw(obs, expected_c=self.cnn.in_channels, device=device)  # type: ignore[attr-defined]
            feats = self.cnn(x)
        else:
            x = _to_tensor(obs, device)
            x = _prep_vector(x, self.obs_dim)  # type: ignore[attr-defined]
            feats = self.backbone(x)  # type: ignore[attr-defined]

        if self.is_discrete:
            logits = self.policy_head(feats)
            return CategoricalDist(logits=logits)
        else:
            mu = self.mu_head(feats)
            return DiagGaussianDist(mean=mu, log_std=self.log_std.expand_as(mu))  # type: ignore[attr-defined]

    def act(self, obs: Tensor | object) -> tuple[Tensor, Tensor]:
        """Sample an action and return it together with its log-probability.

        Parameters
        ----------
        obs :
            Batch of observations as accepted by :meth:`distribution`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple ``(actions, log_probs)`` where both tensors have
            batch dimension ``[B]`` (and additional action dimensions
            for continuous control).
        """
        dist = self.distribution(obs)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp

    def log_prob(self, obs: Tensor | object, actions: Tensor) -> Tensor:
        """Return log-probabilities for ``actions`` under the current policy."""
        return self.distribution(obs).log_prob(actions)

    def entropy(self, obs: Tensor | object) -> Tensor:
        """Return the per-sample entropy of the policy."""
        return self.distribution(obs).entropy()


class ValueNetwork(nn.Module):
    """State-value function :math:`V(s)` with MLP or CNN backbone.

    Parameters
    ----------
    obs_space :
        Observation space of the environment. Must be a ``Box``. Vector
        spaces use an MLP; image-like spaces (rank ≥ 2) use a
        :class:`ConvEncoder`.
    hidden_sizes :
        Hidden-layer sizes for the MLP backbone with vector observations.
    cnn_cfg :
        Optional :class:`ConvEncoderConfig` used for image observations.
        When provided, its ``in_channels`` field is overridden to match
        the number of channels implied by ``obs_space``.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        hidden_sizes: Iterable[int] = (256, 256),
        cnn_cfg: ConvEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("ValueNetwork supports Box observations (vector or images).")

        self.is_image = _is_image_space(obs_space)
        if self.is_image:
            in_ch, in_hw = _infer_c_hw(tuple(int(s) for s in obs_space.shape))
            # Honor user-supplied ConvEncoderConfig; override only in_channels.
            cfg = cnn_cfg if cnn_cfg is not None else ConvEncoderConfig(in_channels=in_ch)
            if int(cfg.in_channels) != int(in_ch):
                cfg = replace(cfg, in_channels=int(in_ch))
            # Pass in_hw so projection is initialized immediately
            self.cnn = ConvEncoder(cfg, in_hw=in_hw)
            self.head = nn.Linear(int(self.cnn.out_dim), 1)
        else:
            self.obs_dim = int(obs_space.shape[0])
            self.net = mlp(self.obs_dim, tuple(hidden_sizes), out_dim=1)

    def forward(self, obs: Tensor | object) -> Tensor:
        """Compute value estimates for a batch of observations.

        Parameters
        ----------
        obs :
            Batch of observations with arbitrary leading dimensions.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``[B]`` containing scalar values.
        """
        device = next(self.parameters()).device
        if self.is_image:
            x = _prep_images_to_nchw(obs, expected_c=self.cnn.in_channels, device=device)  # type: ignore[attr-defined]
            z = self.cnn(x)
            v = self.head(z).squeeze(-1)
            return v
        else:
            x = _to_tensor(obs, device)
            x = _prep_vector(x, self.obs_dim)  # type: ignore[attr-defined]
            v = self.net(x).squeeze(-1)  # type: ignore[attr-defined]
            return v
