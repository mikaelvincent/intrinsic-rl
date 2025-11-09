"""MLP/CNN-based Policy/Value networks for PPO.

Supports:
- **Vector Box** observations: standard MLP backbone.
- **Image Box** observations (rank >= 2): ConvEncoder backbone (NCHW/NHWC tolerant, auto 0..1 scaling).

Discrete actions -> categorical logits. Continuous actions -> diagonal Gaussian with state-independent log_std.

See dev spec §5.1 and Sprint 6 notes for the image pathway.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Union

import gymnasium as gym
import torch
from torch import Tensor, nn

from .distributions import CategoricalDist, DiagGaussianDist
from .layers import mlp  # re-exported for backward compatibility
from .cnn import ConvEncoder, ConvEncoderConfig
from irl.utils.image import preprocess_image, ImagePreprocessConfig


def _space_dims(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Box):
        return int(space.shape[0])
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    raise TypeError(f"Unsupported action/observation space type: {type(space)}")


def _is_image_space(space: gym.Space) -> bool:
    return isinstance(space, gym.spaces.Box) and len(space.shape) >= 2


def _infer_in_channels(shape: tuple[int, ...]) -> int:
    """Return likely channel count from either leading or trailing axis."""
    cand = [int(shape[0]), int(shape[-1])]
    if cand[0] in (1, 3, 4):
        return cand[0]
    return cand[1]


def _to_tensor(x: Tensor | object, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _prep_vector(x: Tensor, obs_dim: int) -> Tensor:
    """Return [N, obs_dim] for vector observations given arbitrary leading dims."""
    t = x
    if t.dim() == 1:
        return t.view(1, obs_dim)
    if t.dim() == 2:
        return t
    # Flatten all leading dims except the last feature dim
    return t.view(-1, obs_dim)


def _prep_images_to_nchw(x: Tensor | object, expected_c: int, device: torch.device) -> Tensor:
    """Return [N, C, H, W] float32 in [0,1] from inputs with arbitrary leading dims.

    Centralized pathway that delegates scaling/layout to `utils.image.preprocess_image`.
    Accepts HWC/CHW/NHWC/NCHW and also collapses extra leading dims (e.g., (T,B,H,W,C)).
    """
    # Convert to a tensor only to inspect rank; avoid forcing float32 here so we retain uint8 scaling semantics.
    xt = x if torch.is_tensor(x) else torch.as_tensor(x)
    if xt.dim() >= 5:
        # Collapse leading dims to [N, H, W, C] or [N, C, H, W]
        xt = xt.reshape(-1, *xt.shape[-3:])
    # Use a shared, explicit preprocessing config: keep channel count as-is (no grayscale), channels-first output.
    cfg = ImagePreprocessConfig(grayscale=False, scale_uint8=True, normalize_mean=None, normalize_std=None, channels_first=True)
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

    Discrete → categorical logits; Box → mean + state-independent log_std.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: Iterable[int] = (256, 256),
        cnn_cfg: ConvEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("PolicyNetwork supports Box observations (vector or images).")

        self.is_image = _is_image_space(obs_space)
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)

        if self.is_image:
            in_ch = _infer_in_channels(tuple(int(s) for s in obs_space.shape))
            # Honor user-supplied ConvEncoderConfig; override only in_channels to match the space.
            cfg = cnn_cfg if cnn_cfg is not None else ConvEncoderConfig(in_channels=in_ch)
            if int(cfg.in_channels) != int(in_ch):
                cfg = replace(cfg, in_channels=int(in_ch))
            self.cnn = ConvEncoder(cfg)
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
        """Sample action and compute its log-prob."""
        dist = self.distribution(obs)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp

    def log_prob(self, obs: Tensor | object, actions: Tensor) -> Tensor:
        return self.distribution(obs).log_prob(actions)

    def entropy(self, obs: Tensor | object) -> Tensor:
        return self.distribution(obs).entropy()


class ValueNetwork(nn.Module):
    """State-value function V(s) with MLP (vector) or CNN (image) backbone."""

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
            in_ch = _infer_in_channels(tuple(int(s) for s in obs_space.shape))
            # Honor user-supplied ConvEncoderConfig; override only in_channels.
            cfg = cnn_cfg if cnn_cfg is not None else ConvEncoderConfig(in_channels=in_ch)
            if int(cfg.in_channels) != int(in_ch):
                cfg = replace(cfg, in_channels=int(in_ch))
            self.cnn = ConvEncoder(cfg)
            self.head = nn.Linear(int(self.cnn.out_dim), 1)
        else:
            self.obs_dim = int(obs_space.shape[0])
            self.net = mlp(self.obs_dim, tuple(hidden_sizes), out_dim=1)

    def forward(self, obs: Tensor | object) -> Tensor:
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
