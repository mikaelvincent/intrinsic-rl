"""Torch-based action distribution wrappers used by PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Categorical, Normal


@dataclass
class CategoricalDist:
    """Categorical distribution with logits."""

    logits: Tensor  # (B, A)

    def sample(self) -> Tensor:
        return Categorical(logits=self.logits).sample()

    def mode(self) -> Tensor:
        return torch.argmax(self.logits, dim=-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        actions = actions.long()
        return Categorical(logits=self.logits).log_prob(actions)

    def entropy(self) -> Tensor:
        return Categorical(logits=self.logits).entropy()


@dataclass
class DiagGaussianDist:
    """Diagonal Gaussian for continuous actions (state‑indep.

    log_std).
    """

    mean: Tensor
    log_std: Tensor

    def _normal(self) -> Normal:
        std = self.log_std.exp()
        return Normal(self.mean, std)

    def sample(self) -> Tensor:
        return self._normal().rsample()  # rsample to enable reparameterization if needed

    def mode(self) -> Tensor:
        return self.mean

    def log_prob(self, actions: Tensor) -> Tensor:
        # Sum over action dimensions -> (B,)
        return self._normal().log_prob(actions).sum(dim=-1)

    def entropy(self) -> Tensor:
        # Sum entropies over action dims -> (B,)
        return self._normal().entropy().sum(dim=-1)
