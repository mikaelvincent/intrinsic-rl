"""Minimal PPO update over minibatches (with reusable optimizers).

Batch must contain observations and actions; `old_log_probs` is optional.
Advantages/targets are 1-D tensors from `irl.algo.advantage.compute_gae`.

New (backward-compatible):
- Optional `optimizers=(pol_opt, val_opt)` parameter allows callers (the
  trainer) to pass persistent Adam optimizers so momentum/state is reused
  across updates and can be checkpointed/resumed.
- Optional `return_stats=True` returns a small dict with `approx_kl`,
  `clip_frac`, `entropy`, `policy_loss`, `value_loss` for logging.

Enhancements:
- Optional **value function clipping** controlled by `cfg.value_clip_range`
  (<=0 disables). Uses the standard PPO formulation with a detached
  baseline `v_old`, i.e. max(MSE(v, vt), MSE(v_clipped, vt)).
- Optional **value loss coefficient** via `cfg.value_coef` (default 0.5).
- Optional **KL penalty** added to the policy loss (`cfg.kl_penalty_coef`)
  and/or **early stop** when the (absolute) approximate KL exceeds
  `cfg.kl_stop`. Both default to disabled (0.0).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, Dict

import torch
from torch import Tensor, nn
from torch.optim import Adam


def _pick(m: Mapping[str, Any], *keys: str, default: Any | None = None) -> Any:
    for k in keys:
        if k in m:
            return m[k]
    return default


def _to_tensor(x: Any, device: torch.device, dtype: torch.dtype | None = None) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype or torch.float32)


def ppo_update(
    policy: Any,
    value: Any,
    batch: Any,
    advantages: Any,
    value_targets: Any,
    cfg: Any,
    *,
    optimizers: Optional[Tuple[Adam, Adam]] = None,
    return_stats: bool = False,
) -> Optional[Dict[str, float]]:
    """Run PPO for several epochs over shuffled minibatches.

    Parameters
    ----------
    policy, value
        Modules with `.distribution(obs)` and `value(obs)`.
    batch
        Mapping with 'obs' (or 'observations') and 'actions', optionally 'old_log_probs'.
    advantages, value_targets
        1-D tensors aligned with flattened observations (N,).
    cfg
        PPOConfig-like with learning_rate, minibatches, epochs, clip_range, entropy_coef.
        Optional: value_coef, value_clip_range, kl_penalty_coef, kl_stop.
    optimizers
        Optional tuple (policy_opt, value_opt). If omitted, fresh Adam optimizers
        are created for this call (keeps unit tests/backward compatibility working).
    return_stats
        If True, returns a dict of logging stats; otherwise returns None.
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
    assert adv_t.shape[0] == N and vtarg_t.shape[0] == N, "adv/targets must match obs count"

    # Normalize advantages once per update (common practice)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

    # Old log-probs: reuse if provided, otherwise compute once and detach
    old_logp = _pick(batch, "old_log_probs")
    if old_logp is None:
        with torch.no_grad():
            dist0 = policy.distribution(obs_t)
            old_logp = dist0.log_prob(act_t)
    old_logp_t = _to_tensor(old_logp, device, dtype=torch.float32).reshape(-1)

    # Cache v_old for value clipping (detached baseline)
    vf_clip = float(getattr(cfg, "value_clip_range", 0.0) or 0.0)
    if vf_clip > 0.0:
        with torch.no_grad():
            v_old_all = value(obs_t).detach()
    else:
        v_old_all = None  # type: ignore[assignment]

    # Optimizers: persistent (preferred) or per-call (backward compatible)
    if optimizers is None:
        pol_opt = Adam(policy.parameters(), lr=float(cfg.learning_rate))
        val_opt = Adam(value.parameters(), lr=float(cfg.learning_rate))
    else:
        pol_opt, val_opt = optimizers

    # Determine minibatch size
    mbs = int(max(1, N // int(max(1, int(cfg.minibatches)))))  # safeguard
    clip_eps = float(cfg.clip_range)
    ent_coef = float(getattr(cfg, "entropy_coef", 0.0))
    val_coef = float(getattr(cfg, "value_coef", 0.5))
    kl_penalty_coef = float(getattr(cfg, "kl_penalty_coef", 0.0))
    kl_stop = float(getattr(cfg, "kl_stop", 0.0))

    # Accumulators for logging
    tot_samples = 0
    sum_entropy = 0.0
    sum_kl = 0.0
    sum_clip_frac = 0.0
    last_pol_loss = 0.0
    last_val_loss = 0.0
    early_stop_triggered = False
    epochs_ran = 0

    for _epoch in range(int(cfg.epochs)):
        epochs_ran += 1
        # Fresh permutation each epoch
        perm = torch.randperm(N, device=device)
        for start in range(0, N, mbs):
            idx = perm[start : start + mbs]
            o = obs_t[idx]
            a = act_t[idx]
            adv = adv_t[idx]
            vt = vtarg_t[idx]
            logp_old = old_logp_t[idx]

            # Policy loss (clipped surrogate) + entropy bonus (+ optional KL penalty)
            dist = policy.distribution(o)
            logp = dist.log_prob(a)
            ratio = (logp - logp_old).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()

            # Approx KL for this minibatch (use abs for penalty/stop)
            approx_kl = (logp_old - logp).mean()
            approx_kl_abs = approx_kl.abs().detach()

            pol_total = policy_loss - ent_coef * entropy
            if kl_penalty_coef > 0.0:
                pol_total = pol_total + float(kl_penalty_coef) * approx_kl_abs

            pol_opt.zero_grad(set_to_none=True)
            pol_total.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            pol_opt.step()

            # Value loss (MSE with optional clipping) * value_coef
            v_pred = value(o)
            if vf_clip > 0.0 and v_old_all is not None:
                v_old = v_old_all[idx]
                v_pred_clipped = v_old + torch.clamp(v_pred - v_old, -vf_clip, vf_clip)
                v_loss_unclipped = (vt - v_pred).pow(2)
                v_loss_clipped = (vt - v_pred_clipped).pow(2)
                v_loss_elem = torch.max(v_loss_unclipped, v_loss_clipped)
            else:
                v_loss_elem = (vt - v_pred).pow(2)
            v_loss = 0.5 * v_loss_elem.mean() * val_coef

            val_opt.zero_grad(set_to_none=True)
            v_loss.backward()
            nn.utils.clip_grad_norm_(value.parameters(), max_norm=1.0)
            val_opt.step()

            # Logging accumulators
            bsz = int(o.shape[0])
            tot_samples += bsz
            sum_entropy += float(entropy.detach().item()) * bsz
            sum_kl += float((logp_old - logp).mean().detach().item()) * bsz  # keep sign for continuity with older logs
            # Fraction of samples where ratio got clipped
            clip_mask = (ratio > (1.0 + clip_eps)) | (ratio < (1.0 - clip_eps))
            sum_clip_frac += float(clip_mask.float().mean().detach().item()) * bsz
            last_pol_loss = float(policy_loss.detach().item())
            last_val_loss = float(v_loss.detach().item())

            # Early stop if KL too large (absolute)
            if kl_stop > 0.0 and float(approx_kl_abs.item()) > kl_stop:
                early_stop_triggered = True
                break  # break minibatch loop

        if early_stop_triggered:
            break  # break epoch loop

    if return_stats:
        denom = max(1, tot_samples)
        return {
            "approx_kl": sum_kl / denom,
            "clip_frac": sum_clip_frac / denom,
            "entropy": sum_entropy / denom,
            "policy_loss": last_pol_loss,
            "value_loss": last_val_loss,
            "early_stop": 1.0 if early_stop_triggered else 0.0,
            "epochs_ran": float(epochs_ran),
        }
    return None
