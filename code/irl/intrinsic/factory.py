"""Factory and helper functions for intrinsic modules.

This module centralises construction, batch computation, and updates for
intrinsic reward modules (ICM, RND, RIDE, RIAC, Proposed).

Proposed ablation variants
--------------------------
Proposed ablations are represented as distinct method strings:

- ``proposed_lp_only``      → Proposed with ``alpha_impact = 0``
- ``proposed_impact_only``  → Proposed with ``alpha_lp = 0``
- ``proposed_nogate``       → Proposed with region gating disabled

These variants are treated as *distinct experiment categories* for plots and
summaries, but are dispatched to the same underlying :class:`Proposed`
implementation.
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

# Canonical intrinsic methods (variants are mapped onto their base method).
_SUPPORTED = {"icm", "rnd", "ride", "riac", "proposed"}

_PROPOSED_VARIANTS = {"proposed_lp_only", "proposed_impact_only", "proposed_nogate"}


def _canonical_method(method: str) -> str:
    """Map experiment-category method names to a canonical implementation key."""
    m = str(method).lower()
    if m.startswith("proposed_"):
        return "proposed"
    return m


def is_intrinsic_method(method: str) -> bool:
    """Return True if the method string corresponds to a supported intrinsic module.

    Notes
    -----
    Proposed ablation variants (``proposed_*``) return True and are routed to the
    Proposed implementation.
    """
    return _canonical_method(str(method)) in _SUPPORTED


def create_intrinsic_module(
    method: str,
    obs_space: gym.Space,
    act_space: Optional[gym.Space],
    device: str | torch.device = "cpu",
    **kwargs: Any,
):
    """Instantiate the intrinsic module for the given method.

    The ``method`` string is treated as an *experiment category* identifier.
    For Proposed ablations (``proposed_*``), this helper returns a Proposed
    module with the corresponding knobs applied.
    """
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
        for k in ("alpha_lp", "region_capacity", "depth_max", "ema_beta_long", "ema_beta_short"):
            if k in kwargs and kwargs[k] is not None:
                riac_kwargs[k] = (
                    float(kwargs[k]) if "alpha" in k or "beta" in k else int(kwargs[k])
                )  # type: ignore[assignment]
        return RIAC(obs_space, act_space, device=device, **riac_kwargs)
    if m == "proposed":
        if act_space is None:
            raise ValueError("Proposed requires an action space (via ICM).")
        prop_kwargs: dict[str, Any] = {}
        # Existing passthroughs
        for k in (
            "alpha_impact",
            "alpha_lp",
            "region_capacity",
            "depth_max",
            "ema_beta_long",
            "ema_beta_short",
            # gating thresholds
            "gate_tau_lp_mult",
            "gate_tau_s",
            "gate_hysteresis_up_mult",
            "gate_min_consec_to_gate",
            "gate_min_regions_for_gating",
        ):
            if k in kwargs and kwargs[k] is not None:
                prop_kwargs[k] = (
                    float(kwargs[k])
                    if ("alpha" in k or "beta" in k or "tau" in k)
                    else int(kwargs[k])
                )  # type: ignore[assignment]

        # Pass through normalization and gating-enable knobs when available.
        # Probe Proposed.__init__ to avoid passing unknown kwargs until the module supports them.
        normalize_inside_val = None
        if "normalize_inside" in kwargs:
            normalize_inside_val = bool(kwargs["normalize_inside"])

        gating_enabled_val = None
        if "gating_enabled" in kwargs:
            gating_enabled_val = bool(kwargs["gating_enabled"])
        elif "gate_enabled" in kwargs:
            # Allow alternate naming from upstream config plumbing.
            gating_enabled_val = bool(kwargs["gate_enabled"])
        elif "gate" in kwargs and isinstance(kwargs["gate"], dict):
            try:
                gating_enabled_val = bool(kwargs["gate"].get("enabled"))
            except Exception:
                pass

        # Apply explicit Proposed ablation semantics from the experiment-category method name.
        if m_raw == "proposed_lp_only":
            prop_kwargs["alpha_impact"] = 0.0
        elif m_raw == "proposed_impact_only":
            prop_kwargs["alpha_lp"] = 0.0
        elif m_raw == "proposed_nogate":
            gating_enabled_val = False
            prop_kwargs["gating_enabled"] = False

        # Introspect Proposed signature to conditionally include constructor kwargs
        try:
            import inspect

            accepted = set(inspect.signature(Proposed.__init__).parameters.keys())
        except Exception:
            accepted = set()

        if normalize_inside_val is not None and "normalize_inside" in accepted:
            prop_kwargs["normalize_inside"] = normalize_inside_val
        if gating_enabled_val is not None and "gating_enabled" in accepted:
            prop_kwargs["gating_enabled"] = gating_enabled_val

        mod = Proposed(obs_space, act_space, device=device, **prop_kwargs)

        # Fallback: if the constructor does not accept ``gating_enabled``, set the
        # attribute directly when present.
        if gating_enabled_val is not None and "gating_enabled" not in accepted:
            try:
                setattr(mod, "gating_enabled", gating_enabled_val)
            except Exception:
                pass
        # Note: we intentionally do NOT mutate outputs_normalized or apply normalize_inside here
        # when the constructor does not expose it to avoid unintended double-normalization.

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
    """Compute unscaled intrinsic rewards for a batch.

    Returns a 1-D tensor of shape ``[N]``.
    """
    m = _canonical_method(str(method))
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
    """Run one or more optimisation steps for the intrinsic module.

    The same batch is reused for ``steps`` updates. A dictionary of scalar
    metrics is returned.
    """
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
    if m == "proposed":
        if actions is None or next_obs is None:
            raise ValueError("Proposed.update requires next_obs and actions.")
        return dict(module.update(obs, next_obs, actions, steps=int(steps)))
    raise ValueError(f"Unsupported intrinsic method for update: {method!r}")
