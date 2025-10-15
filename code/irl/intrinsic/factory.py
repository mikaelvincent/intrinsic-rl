"""Factory and thin helpers for intrinsic modules.

This module provides a small, explicit switchboard for constructing intrinsic
reward modules and interacting with them in a unified way from the training loop.

Supported methods (Sprint 1+2):
- "icm" : Intrinsic Curiosity Module (requires obs + next_obs + actions)
- "rnd" : Random Network Distillation (prefers next_obs; actions unused)
- "ride": Impact-only RIDE; reuses ICM encoder & training (requires next_obs)

Notes
-----
* For "ride", optional kwargs are accepted to wire method-local settings:
    - bin_size: float (episodic embedding binning size)
    - alpha_impact: float (scaling factor for impact reward before global RMS/η)
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import torch

from .icm import ICM
from .rnd import RND
from .ride import RIDE

_SUPPORTED = {"icm", "rnd", "ride"}


def is_intrinsic_method(method: str) -> bool:
    """Return True if the method string corresponds to a supported intrinsic module."""
    return str(method).lower() in _SUPPORTED


def create_intrinsic_module(
    method: str,
    obs_space: gym.Space,
    act_space: Optional[gym.Space],
    device: str | torch.device = "cpu",
    **kwargs: Any,
):
    """Instantiate the intrinsic module for the given method.

    Args:
        method: one of {"icm", "rnd", "ride"}.
        obs_space: Gymnasium observation space.
        act_space: Gymnasium action space (required for ICM and RIDE).
        device: torch device (string or torch.device).
        **kwargs: Optional method-specific settings.
            For "ride":
              - bin_size: float
              - alpha_impact: float

    Returns:
        A module instance (ICM, RND, or RIDE).

    Raises:
        ValueError: if the method is unsupported or required arguments are missing.
    """
    m = str(method).lower()
    if m == "icm":
        if act_space is None:
            raise ValueError("ICM requires an action space.")
        # Ignore ride-specific kwargs if passed by caller
        return ICM(obs_space, act_space, device=device)
    if m == "rnd":
        # Ignore ride-specific kwargs if passed by caller
        return RND(obs_space, device=device)
    if m == "ride":
        if act_space is None:
            # RIDE uses ICM's inverse/forward for training the shared encoder
            raise ValueError("RIDE requires an action space (via ICM).")
        # Only pick supported ride-specific kwargs; ignore the rest for safety
        ride_kwargs: dict[str, Any] = {}
        if "bin_size" in kwargs and kwargs["bin_size"] is not None:
            ride_kwargs["bin_size"] = float(kwargs["bin_size"])
        if "alpha_impact" in kwargs and kwargs["alpha_impact"] is not None:
            ride_kwargs["alpha_impact"] = float(kwargs["alpha_impact"])
        return RIDE(obs_space, act_space, device=device, **ride_kwargs)
    raise ValueError(f"Unsupported intrinsic method: {method!r}")


@torch.no_grad()
def compute_intrinsic_batch(
    module: Any,
    method: str,
    obs: Any,
    next_obs: Any | None,
    actions: Any | None = None,
):
    """Compute *unscaled* intrinsic rewards for a batch.

    Returns a 1-D torch.Tensor of shape [N] (where N is the batch size).
    """
    m = str(method).lower()
    if m == "icm":
        if actions is None or next_obs is None:
            raise ValueError("ICM.compute_batch requires next_obs and actions.")
        return module.compute_batch(obs, next_obs, actions, reduction="none")
    if m == "rnd":
        # Prefer next_obs if provided; actions unused.
        return module.compute_batch(obs, next_obs=next_obs, reduction="none")
    if m == "ride":
        if next_obs is None:
            raise ValueError("RIDE.compute_batch requires next_obs.")
        return module.compute_batch(obs, next_obs, actions=None, reduction="none")
    raise ValueError(f"Unsupported intrinsic method for compute: {method!r}")


def update_module(
    module: Any,
    method: str,
    obs: Any,
    next_obs: Any | None,
    actions: Any | None = None,
    steps: int = 1,
) -> dict:
    """Run one or more optimization steps for the intrinsic module on the same batch.

    Returns a dict with scalar metrics (floats).
    """
    m = str(method).lower()
    if m == "icm":
        if actions is None or next_obs is None:
            raise ValueError("ICM.update requires next_obs and actions.")
        return dict(module.update(obs, next_obs, actions, steps=int(steps)))
    if m == "rnd":
        # Train predictor on preferred target input (next_obs if present).
        x = next_obs if next_obs is not None else obs
        return dict(module.update(x, steps=int(steps)))
    if m == "ride":
        if actions is None or next_obs is None:
            raise ValueError("RIDE.update requires next_obs and actions (for ICM training).")
        return dict(module.update(obs, next_obs, actions, steps=int(steps)))
    raise ValueError(f"Unsupported intrinsic method for update: {method!r}")
