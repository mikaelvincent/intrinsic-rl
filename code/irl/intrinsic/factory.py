"""Factory/helpers for intrinsic modules (ICM, RND, RIDE, RIAC, Proposed).

Unified construction plus compute/update helpers used by the trainer. See devspec/dev_spec_and_plan.md §5 for method
details.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import torch

from .icm import ICM
from .rnd import RND
from .ride import RIDE
from .riac import RIAC
from .proposed import Proposed

_SUPPORTED = {"icm", "rnd", "ride", "riac", "proposed"}


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
    """Instantiate the intrinsic module for the given method."""
    m = str(method).lower()
    if m == "icm":
        if act_space is None:
            raise ValueError("ICM requires an action space.")
        return ICM(obs_space, act_space, device=device)
    if m == "rnd":
        return RND(obs_space, device=device)
    if m == "ride":
        if act_space is None:
            raise ValueError("RIDE requires an action space (via ICM).")
        ride_kwargs: dict[str, Any] = {}
        if "bin_size" in kwargs and kwargs["bin_size"] is not None:
            ride_kwargs["bin_size"] = float(kwargs["bin_size"])
        if "alpha_impact" in kwargs and kwargs["alpha_impact"] is not None:
            ride_kwargs["alpha_impact"] = float(kwargs["alpha_impact"])
        return RIDE(obs_space, act_space, device=device, **ride_kwargs)
    if m == "riac":
        if act_space is None:
            raise ValueError("RIAC requires an action space (via ICM forward model).")
        riac_kwargs: dict[str, Any] = {}
        for k in ("alpha_lp", "region_capacity", "depth_max", "ema_beta_long", "ema_beta_short"):
            if k in kwargs and kwargs[k] is not None:
                riac_kwargs[k] = float(kwargs[k]) if "alpha" in k or "beta" in k else int(kwargs[k])  # type: ignore[assignment]
        return RIAC(obs_space, act_space, device=device, **riac_kwargs)
    if m == "proposed":
        if act_space is None:
            raise ValueError("Proposed requires an action space (via ICM).")
        prop_kwargs: dict[str, Any] = {}
        for k in (
            "alpha_impact",
            "alpha_lp",
            "region_capacity",
            "depth_max",
            "ema_beta_long",
            "ema_beta_short",
            # NEW gating knobs:
            "gate_tau_lp_mult",
            "gate_tau_s",
            "gate_hysteresis_up_mult",
            "gate_min_consec_to_gate",
        ):
            if k in kwargs and kwargs[k] is not None:
                prop_kwargs[k] = float(kwargs[k]) if ("alpha" in k or "beta" in k or "tau" in k) else int(kwargs[k])  # type: ignore[assignment]
        return Proposed(obs_space, act_space, device=device, **prop_kwargs)
    raise ValueError(f"Unsupported intrinsic method: {method!r}")


@torch.no_grad()
def compute_intrinsic_batch(
    module: Any,
    method: str,
    obs: Any,
    next_obs: Any | None,
    actions: Any | None = None,
):
    """Compute unscaled intrinsic rewards for a batch (returns [N])."""
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
    if m == "riac":
        if actions is None or next_obs is None:
            raise ValueError("RIAC.compute_batch requires next_obs and actions.")
        return module.compute_batch(obs, next_obs, actions, reduction="none")
    if m == "proposed":
        if actions is None or next_obs is None:
            raise ValueError("Proposed.compute_batch requires next_obs and actions.")
        return module.compute_batch(obs, next_obs, actions, reduction="none")
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
        x = next_obs if next_obs is not None else obs
        return dict(module.update(x, steps=int(steps)))
    if m == "ride":
        if actions is None or next_obs is None:
            raise ValueError("RIDE.update requires next_obs and actions (for ICM training).")
        return dict(module.update(obs, next_obs, actions, steps=int(steps)))
    if m == "riac":
        if actions is None or next_obs is None:
            raise ValueError("RIAC.update requires next_obs and actions.")
        return dict(module.update(obs, next_obs, actions, steps=int(steps)))
    if m == "proposed":
        if actions is None or next_obs is None:
            raise ValueError("Proposed.update requires next_obs and actions.")
        return dict(module.update(obs, next_obs, actions, steps=int(steps)))
    raise ValueError(f"Unsupported intrinsic method for update: {method!r}")
