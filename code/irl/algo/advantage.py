"""Generalized Advantage Estimation (GAE) implementation.

The function is intentionally tolerant about the batch structure to keep the
PPO backbone decoupled from the data-collection code that will arrive later.

Accepted batch keys (fallbacks in order):
- observations:  "obs", "observations"
- next_observations: "next_obs", "next_observations"  (optional; see below)
- rewards:       "rewards", "r_total", "r"
- dones:         "dones", "terminals", "done"

If `next_observations` are not provided, we compute v_{t+1} by shifting v_t and
bootstrapping the last step with 0. This is safe for terminated transitions
because (1 - done) will nullify the bootstrap term; if you have non-terminal
last steps and need bootstrapping, pass `next_observations`.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import torch
from torch import Tensor, nn


def _pick(m: Mapping[str, Any], *keys: str, default: Any | None = None) -> Any:
    for k in keys:
        if k in m:
            return m[k]
    return default


def _to_tensor(
    x: Any, device: torch.device, dtype: torch.dtype | None = None
) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype or torch.float32)


def compute_gae(
    batch: Any, value_fn: Any, gamma: float, lam: float
) -> Tuple[Tensor, Tensor]:
    """Compute advantages and value targets (TD(λ)/GAE).

    Args:
        batch: mapping-like container with observations, rewards, dones, etc.
        value_fn: callable nn.Module mapping observations -> state values.
        gamma: discount factor in (0, 1].
        lam: GAE lambda in [0, 1].

    Returns:
        (advantages, value_targets) — both flattened to shape (N_total,)
    """
    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping/dict-like object")

    device = next(value_fn.parameters()).device  # type: ignore[attr-defined]

    obs = _pick(batch, "obs", "observations")
    next_obs = _pick(batch, "next_obs", "next_observations")
    rewards = _pick(batch, "rewards", "r_total", "r")
    dones = _pick(batch, "dones", "terminals", "done")

    if obs is None or rewards is None or dones is None:
        raise KeyError("batch must contain observations, rewards and dones.")

    # Convert to tensors on correct device
    obs_t = _to_tensor(obs, device)
    rew_t = _to_tensor(rewards, device, dtype=torch.float32)
    done_t = _to_tensor(dones, device, dtype=torch.float32)

    # Normalize shapes to (T, B, ...) if possible; otherwise treat first dim as time.
    T = obs_t.shape[0]
    B = int(rew_t.numel() // T)
    rew_t = rew_t.view(T, B)
    done_t = done_t.view(T, B)

    # Values for s_t
    with torch.no_grad():
        v_t = value_fn(obs_t).view(T, B)  # type: ignore[misc]

        # Values for s_{t+1}
        if next_obs is not None:
            next_obs_t = _to_tensor(next_obs, device)
            v_tp1 = value_fn(next_obs_t).view(T, B)  # type: ignore[misc]
        else:
            # Shift v_t forward by one step; bootstrap last with zeros (safe if last is terminal)
            v_tp1 = torch.zeros_like(v_t)
            if T > 1:
                v_tp1[:-1] = v_t[1:]
            v_tp1[-1] = 0.0

        # Backward GAE
        adv = torch.zeros_like(v_t)
        last_adv = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            delta = rew_t[t] + gamma * (1.0 - done_t[t]) * v_tp1[t] - v_t[t]
            last_adv = delta + gamma * lam * (1.0 - done_t[t]) * last_adv
            adv[t] = last_adv

        v_targets = adv + v_t

    # Flatten to 1-D for PPO update
    adv = adv.reshape(-1)
    v_targets = v_targets.reshape(-1)
    return adv, v_targets
