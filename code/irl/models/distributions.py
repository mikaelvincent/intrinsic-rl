from __future__ import annotations

from dataclasses import dataclass, field

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


def _atanh(x: Tensor) -> Tensor:
    if hasattr(torch, "atanh"):
        return torch.atanh(x)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


@dataclass
class SquashedDiagGaussianDist:
    mean: Tensor
    log_std: Tensor
    low: Tensor
    high: Tensor
    eps: float = 1e-6

    _scale: Tensor = field(init=False, repr=False)
    _bias: Tensor = field(init=False, repr=False)
    _log_scale_sum: Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dtype = self.mean.dtype
        device = self.mean.device

        low = self.low.to(device=device, dtype=dtype)
        high = self.high.to(device=device, dtype=dtype)

        scale = (high - low) / 2.0
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("SquashedDiagGaussianDist requires finite high>low bounds.")

        self._scale = scale
        self._bias = (high + low) / 2.0
        self._log_scale_sum = torch.log(self._scale).sum()

    def _normal(self) -> Normal:
        return Normal(self.mean, self.log_std.exp())

    def _squash(self, u: Tensor) -> Tensor:
        y = torch.tanh(u)
        return self._bias + self._scale * y

    def sample(self) -> Tensor:
        u = self._normal().rsample()
        return self._squash(u)

    def mode(self) -> Tensor:
        return self._squash(self.mean)

    def log_prob(self, actions: Tensor) -> Tensor:
        a = actions.to(device=self.mean.device, dtype=self.mean.dtype)
        y = (a - self._bias) / self._scale
        y = torch.clamp(y, -1.0 + float(self.eps), 1.0 - float(self.eps))

        u = _atanh(y)
        logp_u = self._normal().log_prob(u).sum(dim=-1)

        one_minus = torch.clamp(1.0 - y.pow(2), min=float(self.eps))
        log_det = torch.log(one_minus).sum(dim=-1)

        return logp_u - log_det - self._log_scale_sum

    def entropy(self) -> Tensor:
        # Exact entropy has no closed form after tanh + affine squashing.
        n = self._normal()
        base = n.entropy().sum(dim=-1)

        u = n.rsample()
        y = torch.tanh(u)
        one_minus = torch.clamp(1.0 - y.pow(2), min=float(self.eps))
        log_det_j = torch.log(one_minus).sum(dim=-1) + self._log_scale_sum

        return base + log_det_j
