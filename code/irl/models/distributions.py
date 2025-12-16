from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical, Normal


@dataclass
class CategoricalDist:
    logits: Tensor

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
    mean: Tensor
    log_std: Tensor

    def _normal(self) -> Normal:
        return Normal(self.mean, self.log_std.exp())

    def sample(self) -> Tensor:
        return self._normal().rsample()

    def mode(self) -> Tensor:
        return self.mean

    def log_prob(self, actions: Tensor) -> Tensor:
        return self._normal().log_prob(actions).sum(dim=-1)

    def entropy(self) -> Tensor:
        return self._normal().entropy().sum(dim=-1)
