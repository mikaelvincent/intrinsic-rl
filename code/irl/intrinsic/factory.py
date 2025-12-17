from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import torch

from .icm import ICM
from .glpe import GLPE
from .riac import RIAC
from .ride import RIDE
from .rnd import RND

_SUPPORTED = {"icm", "rnd", "ride", "riac", "glpe"}


def _canonical_method(method: str) -> str:
    m = str(method).lower()
    if m.startswith("glpe_"):
        return "glpe"
    return m


def is_intrinsic_method(method: str) -> bool:
    return _canonical_method(str(method)) in _SUPPORTED


def create_intrinsic_module(
    method: str,
    obs_space: gym.Space,
    act_space: Optional[gym.Space],
    device: str | torch.device = "cpu",
    **kwargs: Any,
):
    m_raw = str(method).lower()
    m = _canonical_method(m_raw)

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
        for k in (
            "alpha_lp",
            "region_capacity",
            "depth_max",
            "ema_beta_long",
            "ema_beta_short",
            "checkpoint_include_points",
        ):
            if k in kwargs and kwargs[k] is not None:
                if k == "checkpoint_include_points":
                    riac_kwargs[k] = bool(kwargs[k])
                else:
                    riac_kwargs[k] = (
                        float(kwargs[k]) if ("alpha" in k or "beta" in k) else int(kwargs[k])
                    )
        return RIAC(obs_space, act_space, device=device, **riac_kwargs)
    if m == "glpe":
        if act_space is None:
            raise ValueError("GLPE requires an action space (via ICM).")
        glpe_kwargs: dict[str, Any] = {}
        for k in (
            "alpha_impact",
            "alpha_lp",
            "region_capacity",
            "depth_max",
            "ema_beta_long",
            "ema_beta_short",
            "gate_tau_lp_mult",
            "gate_tau_s",
            "gate_hysteresis_up_mult",
            "gate_min_consec_to_gate",
            "gate_min_regions_for_gating",
            "checkpoint_include_points",
        ):
            if k in kwargs and kwargs[k] is not None:
                if k == "checkpoint_include_points":
                    glpe_kwargs[k] = bool(kwargs[k])
                else:
                    glpe_kwargs[k] = (
                        float(kwargs[k])
                        if ("alpha" in k or "beta" in k or "tau" in k)
                        else int(kwargs[k])
                    )

        normalize_inside_val = None
        if "normalize_inside" in kwargs:
            normalize_inside_val = bool(kwargs["normalize_inside"])

        gating_enabled_val = None
        if "gating_enabled" in kwargs:
            gating_enabled_val = bool(kwargs["gating_enabled"])
        elif "gate_enabled" in kwargs:
            gating_enabled_val = bool(kwargs["gate_enabled"])
        elif "gate" in kwargs and isinstance(kwargs["gate"], dict):
            try:
                gating_enabled_val = bool(kwargs["gate"].get("enabled"))
            except Exception:
                pass

        if m_raw == "glpe_lp_only":
            glpe_kwargs["alpha_impact"] = 0.0
        elif m_raw == "glpe_impact_only":
            glpe_kwargs["alpha_lp"] = 0.0
        elif m_raw == "glpe_nogate":
            gating_enabled_val = False
            glpe_kwargs["gating_enabled"] = False

        try:
            import inspect

            accepted = set(inspect.signature(GLPE.__init__).parameters.keys())
        except Exception:
            accepted = set()

        if normalize_inside_val is not None and "normalize_inside" in accepted:
            glpe_kwargs["normalize_inside"] = normalize_inside_val
        if gating_enabled_val is not None and "gating_enabled" in accepted:
            glpe_kwargs["gating_enabled"] = gating_enabled_val

        mod = GLPE(obs_space, act_space, device=device, **glpe_kwargs)

        if gating_enabled_val is not None and "gating_enabled" not in accepted:
            try:
                setattr(mod, "gating_enabled", gating_enabled_val)
            except Exception:
                pass

        return mod
    raise ValueError(f"Unsupported intrinsic method: {method!r}")


@torch.no_grad()
def compute_intrinsic_batch(
    module: Any,
    method: str,
    obs: Any,
    next_obs: Any | None,
    actions: Any | None = None,
):
    m = _canonical_method(str(method))
    if m == "icm":
        if actions is None or next_obs is None:
            raise ValueError("ICM.compute_batch requires next_obs and actions.")
        return module.compute_batch(obs, next_obs, actions, reduction="none")
    if m == "rnd":
        return module.compute_batch(obs, next_obs=next_obs, reduction="none")
    if m == "ride":
        if next_obs is None:
            raise ValueError("RIDE.compute_batch requires next_obs.")
        return module.compute_batch(obs, next_obs, actions=None, reduction="none")
    if m == "riac":
        if actions is None or next_obs is None:
            raise ValueError("RIAC.compute_batch requires next_obs and actions.")
        return module.compute_batch(obs, next_obs, actions, reduction="none")
    if m == "glpe":
        if actions is None or next_obs is None:
            raise ValueError("GLPE.compute_batch requires next_obs and actions.")
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
    m = _canonical_method(str(method))
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
    if m == "glpe":
        if actions is None or next_obs is None:
            raise ValueError("GLPE.update requires next_obs and actions.")
        return dict(module.update(obs, next_obs, actions, steps=int(steps)))
    raise ValueError(f"Unsupported intrinsic method for update: {method!r}")
