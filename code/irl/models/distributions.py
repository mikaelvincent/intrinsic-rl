"""Thin action-distribution wrappers used by PPO.

These classes provide a small, torch-native surface around
:mod:`torch.distributions` so the rest of the codebase does not depend
on the distribution types directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple  # noqa: F401  (kept for compatibility / future use)

import torch
from torch import Tensor
from torch.distributions import Categorical, Normal


@dataclass
class CategoricalDist:
    """Categorical policy distribution parameterised by logits.

    Parameters
    ----------
    logits :
        Unnormalised log-probabilities with shape ``(B, A)`` where ``B``
        is the batch size and ``A`` the number of discrete actions.
    """

    logits: Tensor  # (B, A)

    def sample(self) -> Tensor:
        """Draw a categorical sample for each batch element."""
        return Categorical(logits=self.logits).sample()

    def mode(self) -> Tensor:
        """Return the most probable action index for each batch element."""
        return torch.argmax(self.logits, dim=-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        """Return log-probabilities for `actions`.

        Parameters
        ----------
        actions :
            Integer action indices with shape broadcastable to ``(B,)``.

        Returns
        -------
        torch.Tensor
            Log-probabilities with shape ``(B,)``.
        """
        actions = actions.long()
        return Categorical(logits=self.logits).log_prob(actions)

    def entropy(self) -> Tensor:
        """Return the per-sample entropy of the distribution."""
        return Categorical(logits=self.logits).entropy()


@dataclass
class DiagGaussianDist:
    """Diagonal Gaussian distribution for continuous actions.

    Parameters
    ----------
    mean :
        Mean action vector with shape ``(B, A)``.
    log_std :
        Log standard deviation with shape compatible with ``mean``.
        In typical use this is broadcast from a single ``(A,)`` vector
        across the batch.
    """

    mean: Tensor
    log_std: Tensor

    def _normal(self) -> Normal:
        """Return the underlying :class:`Normal` distribution."""
        std = self.log_std.exp()
        return Normal(self.mean, std)

    def sample(self) -> Tensor:
        """Draw a reparameterised sample (``rsample``) for each batch element."""
        return self._normal().rsample()

    def mode(self) -> Tensor:
        """Return the mean action for each batch element."""
        return self.mean

    def log_prob(self, actions: Tensor) -> Tensor:
        """Return log-probabilities for continuous actions.

        Parameters
        ----------
        actions :
            Continuous action tensor with the same shape as ``mean``.

        Returns
        -------
        torch.Tensor
            Log-probabilities summed over action dimensions, with shape
            ``(B,)``.
        """
        # Sum over action dimensions -> (B,)
        return self._normal().log_prob(actions).sum(dim=-1)

    def entropy(self) -> Tensor:
        """Return the entropy of the diagonal Gaussian per batch element."""
        # Sum entropies over action dims -> (B,)
        return self._normal().entropy().sum(dim=-1)
