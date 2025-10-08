"""Policy/Value networks for PPO (MLP-based).

Notes:
- Observations are flattened before feeding into MLPs.
- Supports Discrete and Box action spaces.
- Continuous actions use diagonal Gaussian with state-independent log_std.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

import gymnasium as gym
import torch
from torch import Tensor, nn

from .distributions import CategoricalDist, DiagGaussianDist


# -------------------- small helpers --------------------


class FlattenObs(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x.view(1, -1)
        return x.view(x.size(0), -1)


def mlp(
    in_dim: int, hidden: Sequence[int] = (256, 256), out_dim: int | None = None
) -> nn.Sequential:
    layers: list[nn.Module] = [FlattenObs()]
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
        last = h
    if out_dim is not None:
        layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def _space_dims(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Box):
        return int(space.shape[0])
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    raise TypeError(f"Unsupported action/observation space type: {type(space)}")


# -------------------- Networks --------------------


class PolicyNetwork(nn.Module):
    """PPO policy network.

    For Discrete actions: categorical logits head.
    For Box actions: mean head + state-independent log_std parameter.

    Args:
        obs_space: Gymnasium observation space (expects flat-able arrays).
        action_space: Discrete or Box action space.
        hidden_sizes: MLP widths.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: Iterable[int] = (256, 256),
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            # For step 5, we assume vector observations (images/CNN arrive in later sprints).
            raise TypeError(
                "PolicyNetwork currently supports Box observations. "
                "Image encoders will be added in a later sprint."
            )

        self.obs_dim = int(obs_space.shape[0])
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)

        self.backbone = mlp(self.obs_dim, tuple(hidden_sizes), out_dim=None)

        if self.is_discrete:
            self.n_actions = int(_space_dims(action_space))
            self.policy_head = nn.Linear(hidden_sizes[-1], self.n_actions)
            # No additional parameters needed
        else:
            if not isinstance(action_space, gym.spaces.Box):
                raise TypeError(f"Unsupported action space: {type(action_space)}")
            self.act_dim = int(_space_dims(action_space))
            self.mu_head = nn.Linear(hidden_sizes[-1], self.act_dim)
            # State-independent log_std (learned)
            self.log_std = nn.Parameter(torch.zeros(self.act_dim))

    # ----- API -----

    def distribution(self, obs: Tensor) -> Union[CategoricalDist, DiagGaussianDist]:
        x = obs
        if not torch.is_tensor(x):
            x = torch.as_tensor(
                x,
                dtype=torch.float32,
                device=self.log_std.device if hasattr(self, "log_std") else next(self.parameters()).device,  # type: ignore[union-attr]
            )
        else:
            x = x.to(device=self.log_std.device if hasattr(self, "log_std") else next(self.parameters()).device)  # type: ignore[union-attr]
        # Expect 2D [batch, obs_dim]; callers should reshape if needed.
        if x.dim() > 2:
            x = x.view(-1, self.obs_dim)
        feats = self.backbone(x)
        if self.is_discrete:
            logits = self.policy_head(feats)
            return CategoricalDist(logits=logits)
        else:
            mu = self.mu_head(feats)
            return DiagGaussianDist(mean=mu, log_std=self.log_std.expand_as(mu))

    def act(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Sample action and compute its log-prob."""
        dist = self.distribution(obs)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp

    def log_prob(self, obs: Tensor, actions: Tensor) -> Tensor:
        return self.distribution(obs).log_prob(actions)

    def entropy(self, obs: Tensor) -> Tensor:
        return self.distribution(obs).entropy()


class ValueNetwork(nn.Module):
    """State-value network (MLP -> scalar)."""

    def __init__(self, obs_space: gym.Space, hidden_sizes: Iterable[int] = (256, 256)) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(
                "ValueNetwork currently supports Box observations. "
                "Image encoders will be added in a later sprint."
            )
        self.obs_dim = int(obs_space.shape[0])
        self.net = mlp(self.obs_dim, tuple(hidden_sizes), out_dim=1)

    def forward(self, obs: Tensor) -> Tensor:
        # Accept [N, obs_dim] or [T, B, obs_dim]; reshape to 2D batch then restore 1D outputs.
        x = obs
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        else:
            x = x.to(device=next(self.parameters()).device, dtype=torch.float32)
        if x.dim() > 2:
            x = x.view(-1, self.obs_dim)
        v = self.net(x).squeeze(-1)
        return v
