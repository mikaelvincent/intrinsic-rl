"""PPO optimization loop (minimal implementation).

This function expects a "batch" mapping with at least:
- "obs" / "observations":      (N, obs_dim)
- "actions":                   (N,) for Discrete or (N, act_dim) for Box
- optionally "old_log_probs":  (N,)  -> if absent, computed once at epoch 0

`advantages` and `value_targets` should be 1-D tensors of length N, as returned
by `irl.algo.advantage.compute_gae`.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.optim import Adam


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


def ppo_update(
    policy: Any, value: Any, batch: Any, advantages: Any, value_targets: Any, cfg: Any
) -> None:
    """Run PPO updates for several epochs over minibatches.

    The function creates lightweight Adam optimizers internally using the
    learning rate from `cfg`. In later sprints, higher-level training code may
    externalize optimizers, schedules, and logging.

    Args:
        policy: policy network (nn.Module) providing a `distribution(obs)` method.
        value: value network (nn.Module) mapping obs -> scalar value.
        batch: mapping with "obs"/"actions" (and optional "old_log_probs").
        advantages: 1-D tensor of length N.
        value_targets: 1-D tensor of length N.
        cfg: PPOConfig-like object with fields:
             epochs, minibatches, learning_rate, clip_range, entropy_coef.
    """
    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping/dict-like object")

    device = next(policy.parameters()).device  # type: ignore[attr-defined]

    obs = _pick(batch, "obs", "observations")
    actions = _pick(batch, "actions")
    if obs is None or actions is None:
        raise KeyError("batch must contain 'obs' (or 'observations') and 'actions'.")

    obs_t = _to_tensor(obs, device)
    act_t = _to_tensor(actions, device)
    adv_t = _to_tensor(advantages, device, dtype=torch.float32).reshape(-1)
    vtarg_t = _to_tensor(value_targets, device, dtype=torch.float32).reshape(-1)

    N = obs_t.shape[0]
    assert (
        adv_t.shape[0] == N and vtarg_t.shape[0] == N
    ), "adv/targets must match obs count"

    # Normalize advantages once per update (common practice)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

    # Old log-probs: reuse if provided, otherwise compute once and detach
    old_logp = _pick(batch, "old_log_probs")
    if old_logp is None:
        with torch.no_grad():
            dist0 = policy.distribution(obs_t)
            old_logp = dist0.log_prob(act_t)
    old_logp_t = _to_tensor(old_logp, device, dtype=torch.float32).reshape(-1)

    # Optimizers (kept local for now)
    pol_opt = Adam(policy.parameters(), lr=float(cfg.learning_rate))
    val_opt = Adam(value.parameters(), lr=float(cfg.learning_rate))

    # Determine minibatch size
    mbs = int(max(1, N // int(max(1, int(cfg.minibatches)))))  # safe guard
    clip_eps = float(cfg.clip_range)
    ent_coef = float(getattr(cfg, "entropy_coef", 0.0))

    for _ in range(int(cfg.epochs)):
        # Fresh permutation each epoch
        perm = torch.randperm(N, device=device)
        for start in range(0, N, mbs):
            idx = perm[start : start + mbs]
            o = obs_t[idx]
            a = act_t[idx]
            adv = adv_t[idx]
            vt = vtarg_t[idx]
            logp_old = old_logp_t[idx]

            # Policy loss (clipped surrogate) + entropy bonus
            dist = policy.distribution(o)
            logp = dist.log_prob(a)
            ratio = (logp - logp_old).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()
            pol_total = policy_loss - ent_coef * entropy

            pol_opt.zero_grad(set_to_none=True)
            pol_total.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            pol_opt.step()

            # Value loss (MSE)
            v_pred = value(o)
            v_loss = 0.5 * (vt - v_pred).pow(2).mean()

            val_opt.zero_grad(set_to_none=True)
            v_loss.backward()
            nn.utils.clip_grad_norm_(value.parameters(), max_norm=1.0)
            val_opt.step()
