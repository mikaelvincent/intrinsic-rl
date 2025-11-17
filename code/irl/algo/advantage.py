"""Generalized Advantage Estimation with shape normalization and timeout-aware masking.

Common batch aliases are accepted, observations are reshaped to time-major
layouts, and rewards/dones are normalized to (T, B). Optional masks separate
true terminals from truncations, with configurable bootstrapping behavior.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple, Optional

import torch
from torch import Tensor, nn


def _pick(m: Mapping[str, Any], *keys: str, default: Any | None = None) -> Any:
    for k in keys:
        if k in m:
            return m[k]
    return default


def _to_tensor(x: Any, device: torch.device, dtype: torch.dtype | None = None) -> Tensor:
    """Convert to a tensor on `device`, preserving dtype unless explicitly requested.

    Note: We *do not* force float32 by default to avoid breaking image pipelines
    that rely on integer inputs (uint8) to trigger scaling in preprocessors.
    """
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype or x.dtype)
    if dtype is None:
        return torch.as_tensor(x, device=device)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _ensure_time_batch_layout(
    obs_t: Tensor,
    next_obs_t: Tensor | None,
    rew_t: Tensor,
    done_t: Tensor,
) -> tuple[Tensor, Tensor | None, Tensor, Tensor, int, int]:
    """Return tensors with rewards/dones in (T,B) and obs/next_obs time-major.

    Rules:
    - If rewards/dones are 2-D, we trust those dims to define (T,B) (order checked).
    - If rewards/dones are 1-D, we infer T,B from obs' first two dims if present; otherwise B=1.
    - If obs are (B,T,...) while (T,B) is required, auto-swap the first two axes.
    """
    # Normalize rewards/dones shapes to 1-D or 2-D tensors on same device/dtype
    dev = obs_t.device
    rew_t = rew_t.to(dev, dtype=torch.float32)
    done_t = done_t.to(dev, dtype=torch.float32)

    # Determine (T,B) from rewards/dones when 2-D, or infer from obs otherwise
    if rew_t.dim() == 2 and done_t.dim() == 2:
        rT, rB = int(rew_t.size(0)), int(rew_t.size(1))
        # Try to align obs' leading dims with (T,B) (or (B,T) -> swap)
        if obs_t.dim() >= 2:
            o0, o1 = int(obs_t.size(0)), int(obs_t.size(1))
            if o0 == rB and o1 == rT:
                # batch-major provided; swap to time-major
                obs_t = obs_t.transpose(0, 1)
                if next_obs_t is not None:
                    next_obs_t = next_obs_t.transpose(0, 1)
            elif o0 != rT or (obs_t.dim() >= 2 and o1 != rB):
                # If obs has only one leading (T) dim (e.g., B==1), tolerate o1 mismatch.
                if not (o0 == rT and (obs_t.dim() == 1 or rB == 1)):
                    raise ValueError(
                        "compute_gae: inconsistent shapes — expected obs leading dims to match "
                        f"(T,B)=({rT},{rB}) (allowing B==1), got obs[0:2]={tuple(obs_t.shape[:2])} "
                        f"and rewards shape={tuple(rew_t.shape)}."
                    )
        # Ensure rewards/dones are exactly (T,B)
        T, B = rT, rB
    else:
        # Flatten rewards/dones to 1-D then infer (T,B).
        N = int(rew_t.numel())
        if N != int(done_t.numel()):
            raise ValueError(
                "compute_gae: rewards and dones must have the same number of elements: "
                f"{N} vs {int(done_t.numel())}."
            )
        if obs_t.dim() >= 2:
            o0 = int(obs_t.size(0))
            o1 = int(obs_t.size(1))
            if o0 * o1 == N:
                T, B = o0, o1
            else:
                # Ambiguous; fall back to B=1 if compatible
                if o0 == N:
                    T, B = o0, 1
                else:
                    raise ValueError(
                        "compute_gae: cannot infer (T,B) from 1-D rewards/dones and obs shape. "
                        f"Got N={N}, obs leading dims={tuple(obs_t.shape[:2])}."
                    )
        else:
            # Only a time axis available → treat as a single environment (B=1)
            o0 = int(obs_t.size(0))
            if o0 != N:
                raise ValueError(
                    "compute_gae: rewards/dones length does not match obs time length. "
                    f"N={N}, T(obs)={o0}."
                )
            T, B = o0, 1
        # Reshape rewards/dones to (T,B)
        rew_t = rew_t.reshape(T, B)
        done_t = done_t.reshape(T, B)

    # If obs are batch-major (B,T,...), swap to time-major
    if obs_t.dim() >= 2 and (int(obs_t.size(0)) == B and int(obs_t.size(1)) == T):
        obs_t = obs_t.transpose(0, 1)
        if next_obs_t is not None:
            next_obs_t = next_obs_t.transpose(0, 1)

    # Final sanity: first two dims must be (T,B) if present
    if obs_t.dim() >= 2:
        if not (int(obs_t.size(0)) == T and (B == 1 or int(obs_t.size(1)) == B)):
            raise ValueError(
                "compute_gae: observation tensor is not time-major (T,B,...). "
                f"Expected leading dims (T,B)=({T},{B}); got {tuple(obs_t.shape[:2])}."
            )

    return obs_t, next_obs_t, rew_t, done_t, T, B


def _coerce_mask_to_TB(mask: Tensor, T: int, B: int, like_done: Tensor) -> Tensor:
    """Best-effort reshape/transpose of a mask to (T,B) float32 on the same device as like_done."""
    dev = like_done.device
    m = mask.to(dev, dtype=torch.float32)
    if m.dim() == 2:
        if (int(m.size(0)), int(m.size(1))) == (T, B):
            return m
        if (int(m.size(0)), int(m.size(1))) == (B, T):
            return m.transpose(0, 1)
        if int(m.numel()) == T * B:
            return m.reshape(T, B)
        raise ValueError(
            f"compute_gae: cannot align mask of shape {tuple(m.shape)} to (T,B)=({T},{B})."
        )
    if m.dim() == 1:
        if int(m.numel()) == T * B:
            return m.reshape(T, B)
        if int(m.numel()) == T and B == 1:
            return m.view(T, 1)
        if int(m.numel()) == B and T == 1:
            return m.view(1, B)
        raise ValueError(
            f"compute_gae: 1-D mask length {int(m.numel())} incompatible with T*B={T*B}."
        )
    raise ValueError(f"compute_gae: mask must be 1-D or 2-D, got shape {tuple(m.shape)}.")


def compute_gae(
    batch: Any,
    value_fn: Any,
    gamma: float,
    lam: float,
    *,
    bootstrap_on_timeouts: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Return (advantages, value_targets) flattened to (N,).

    `batch` may use aliases like "obs"/"observations", "dones"/"done", etc.
    Shapes:
      * Observations: (T,B,...) or (B,T,...) — auto-swapped to time-major; B can be 1.
      * rewards/dones: (T,B) or 1-D length N=T*B — reshaped to (T,B) internally.

    Timeout-aware handling
    ----------------------
    You can optionally provide **separate** masks for terminals and truncations/timeouts:
      - terminals: keys accepted: "terminals", "terminal", "terms"
      - truncations/timeouts: keys accepted: "truncations", "timeouts", "truncated", "time_limits"

    The effective 'done' used in GAE is:
      done_eff = terminals OR ( (not bootstrap_on_timeouts) AND truncations )

    Defaults preserve current behavior (treat truncations as done).
    """
    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping/dict-like object")

    device = next(value_fn.parameters()).device  # type: ignore[attr-defined]

    obs = _pick(batch, "obs", "observations")
    next_obs = _pick(batch, "next_obs", "next_observations")
    rewards = _pick(batch, "rewards", "r_total", "r")
    # Primary 'dones' fallback; terminals/truncations may refine it
    dones_any = _pick(batch, "dones", "done", "terminals")  # keep "terminals" as a loose alias
    terminals_raw = _pick(batch, "terminals", "terminal", "terms")
    trunc_raw = _pick(batch, "truncations", "timeouts", "truncated", "time_limits")

    if obs is None or rewards is None:
        raise KeyError("batch must contain observations and rewards.")
    if dones_any is None and terminals_raw is None and trunc_raw is None:
        raise KeyError("batch must contain 'dones' or separate 'terminals'/'truncations' masks.")

    # Convert to tensors on correct device
    obs_t = _to_tensor(obs, device)
    next_obs_t = None if next_obs is None else _to_tensor(next_obs, device)
    rew_t = _to_tensor(rewards, device, dtype=torch.float32)

    # If a coarse 'dones' is provided, use it to infer layout; refine later if separate masks exist
    done_base = torch.zeros_like(rew_t, dtype=torch.float32)
    if dones_any is not None:
        done_base = _to_tensor(dones_any, device, dtype=torch.float32)

    # Canonicalize shapes/layout to time-major (T,B,...) and (T,B)
    obs_t, next_obs_t, rew_t, done_t, T, B = _ensure_time_batch_layout(
        obs_t, next_obs_t, rew_t, done_base
    )

    # Prepare effective done mask (terminals vs timeouts)
    term_t: Optional[Tensor] = None
    trunc_t: Optional[Tensor] = None

    if terminals_raw is not None:
        term_t = _coerce_mask_to_TB(_to_tensor(terminals_raw, device), T, B, done_t)
    if trunc_raw is not None:
        trunc_t = _coerce_mask_to_TB(_to_tensor(trunc_raw, device), T, B, done_t)

    if term_t is None and trunc_t is None:
        done_eff = done_t  # fall back to whatever caller provided
    else:
        term = term_t if term_t is not None else torch.zeros_like(done_t)
        if trunc_t is None:
            done_eff = term
        else:
            if bootstrap_on_timeouts:
                done_eff = term  # ignore truncations in 'done'
            else:
                done_eff = torch.clamp(term + trunc_t, max=1.0)

    # Values for s_t and s_{t+1}
    with torch.no_grad():
        v_t = value_fn(obs_t).view(T, B)  # type: ignore[misc]

        if next_obs_t is not None:
            v_tp1 = value_fn(next_obs_t).view(T, B)  # type: ignore[misc]
        else:
            # Shift v_t forward by one step; bootstrap last with zeros (safe if last is terminal)
            v_tp1 = torch.zeros_like(v_t)
            if T > 1:
                v_tp1[:-1] = v_t[1:]
            v_tp1[-1] = 0.0

        # Backward GAE using the *effective* done mask
        adv = torch.zeros_like(v_t)
        last_adv = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            delta = rew_t[t] + gamma * (1.0 - done_eff[t]) * v_tp1[t] - v_t[t]
            last_adv = delta + gamma * lam * (1.0 - done_eff[t]) * last_adv
            adv[t] = last_adv

        v_targets = adv + v_t

    # Flatten to 1-D for PPO update
    adv = adv.reshape(-1)
    v_targets = v_targets.reshape(-1)
    return adv, v_targets
