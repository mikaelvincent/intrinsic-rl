from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Tuple, Union

import gymnasium as gym
import torch
from torch import Tensor, nn

from irl.utils.image import ImagePreprocessConfig, preprocess_image

from .cnn import ConvEncoder, ConvEncoderConfig
from .distributions import CategoricalDist, DiagGaussianDist
from .layers import mlp


def _space_dims(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Box):
        return int(space.shape[0])
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    raise TypeError(f"Unsupported action/observation space type: {type(space)}")


def _is_image_space(space: gym.Space) -> bool:
    return isinstance(space, gym.spaces.Box) and len(space.shape) >= 2


def _infer_c_hw(shape: tuple[int, ...]) -> Tuple[int, Tuple[int, int]]:
    if len(shape) != 3:
        return shape[-1], (shape[0], shape[1]) if len(shape) >= 2 else (0, 0)

    c0 = shape[0]
    c2 = shape[-1]
    if c0 in (1, 3, 4) and c2 not in (1, 3, 4):
        return c0, (shape[1], shape[2])
    return c2, (shape[0], shape[1])


def _to_tensor(x: Tensor | object, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _prep_vector(x: Tensor, obs_dim: int) -> Tensor:
    t = x
    if t.dim() == 1:
        return t.view(1, obs_dim)
    if t.dim() == 2:
        return t
    return t.view(-1, obs_dim)


def _prep_images_to_nchw(x: Tensor | object, expected_c: int, device: torch.device) -> Tensor:
    xt = x if torch.is_tensor(x) else torch.as_tensor(x)
    if xt.dim() >= 5:
        xt = xt.reshape(-1, *xt.shape[-3:])

    cfg = ImagePreprocessConfig(
        grayscale=False,
        scale_uint8=True,
        normalize_mean=None,
        normalize_std=None,
        channels_first=True,
    )
    out = preprocess_image(xt, cfg=cfg, device=device)

    if out.dim() == 3:
        out = out.unsqueeze(0)

    c = int(out.shape[1])
    if c != int(expected_c):
        raise ValueError(
            f"Image channel mismatch: model expects C={expected_c}, but preprocessed input has C={c}. "
            "Ensure the observation space shape matches the configured encoder channels."
        )
    return out


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        self_obs_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: Iterable[int] = (256, 256),
        cnn_cfg: ConvEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        obs_space = self_obs_space

        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("PolicyNetwork supports Box observations (vector or images).")

        self.is_image = _is_image_space(obs_space)
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)

        if self.is_image:
            in_ch, in_hw = _infer_c_hw(tuple(int(s) for s in obs_space.shape))
            cfg = cnn_cfg if cnn_cfg is not None else ConvEncoderConfig(in_channels=in_ch)
            if int(cfg.in_channels) != int(in_ch):
                cfg = replace(cfg, in_channels=int(in_ch))
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

    def distribution(self, obs: Tensor | object) -> Union[CategoricalDist, DiagGaussianDist]:
        device = self.log_std.device if hasattr(self, "log_std") else next(self.parameters()).device
        if self.is_image:
            x = _prep_images_to_nchw(obs, expected_c=self.cnn.in_channels, device=device)
            feats = self.cnn(x)
        else:
            x = _to_tensor(obs, device)
            x = _prep_vector(x, self.obs_dim)
            feats = self.backbone(x)

        if self.is_discrete:
            logits = self.policy_head(feats)
            return CategoricalDist(logits=logits)
        mu = self.mu_head(feats)
        return DiagGaussianDist(mean=mu, log_std=self.log_std.expand_as(mu))

    def act(self, obs: Tensor | object) -> tuple[Tensor, Tensor]:
        dist = self.distribution(obs)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp

    def log_prob(self, obs: Tensor | object, actions: Tensor) -> Tensor:
        return self.distribution(obs).log_prob(actions)

    def entropy(self, obs: Tensor | object) -> Tensor:
        return self.distribution(obs).entropy()


class ValueNetwork(nn.Module):
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
            cfg = cnn_cfg if cnn_cfg is not None else ConvEncoderConfig(in_channels=in_ch)
            if int(cfg.in_channels) != int(in_ch):
                cfg = replace(cfg, in_channels=int(in_ch))
            self.cnn = ConvEncoder(cfg, in_hw=in_hw)
            self.head = nn.Linear(int(self.cnn.out_dim), 1)
        else:
            self.obs_dim = int(obs_space.shape[0])
            self.net = mlp(self.obs_dim, tuple(hidden_sizes), out_dim=1)

    def forward(self, obs: Tensor | object) -> Tensor:
        device = next(self.parameters()).device
        if self.is_image:
            x = _prep_images_to_nchw(obs, expected_c=self.cnn.in_channels, device=device)
            z = self.cnn(x)
            v = self.head(z).squeeze(-1)
            return v
        x = _to_tensor(obs, device)
        x = _prep_vector(x, self.obs_dim)
        v = self.net(x).squeeze(-1)
        return v
