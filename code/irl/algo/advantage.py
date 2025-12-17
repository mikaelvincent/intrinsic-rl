from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from irl.utils.collections import pick as _pick
from irl.utils.tensors import to_tensor as _to_tensor


def _ensure_time_batch_layout(
    obs_t: Tensor,
    next_obs_t: Tensor | None,
    rew_t: Tensor,
    done_t: Tensor,
) -> tuple[Tensor, Tensor | None, Tensor, Tensor, int, int]:
    dev = obs_t.device
    rew_t = rew_t.to(dev, dtype=torch.float32)
    done_t = done_t.to(dev, dtype=torch.float32)

    if rew_t.dim() == 2 and done_t.dim() == 2:
        rT, rB = int(rew_t.size(0)), int(rew_t.size(1))
        if obs_t.dim() >= 2:
            o0, o1 = int(obs_t.size(0)), int(obs_t.size(1))
            if o0 == rB and o1 == rT:
                obs_t = obs_t.transpose(0, 1)
                if next_obs_t is not None:
                    next_obs_t = next_obs_t.transpose(0, 1)
            elif o0 != rT or (obs_t.dim() >= 2 and o1 != rB):
                if not (o0 == rT and (obs_t.dim() == 1 or rB == 1)):
                    raise ValueError(
                        "compute_gae: inconsistent shapes — expected obs leading dims to match "
                        f"(T,B)=({rT},{rB}) (allowing B==1), got obs[0:2]={tuple(obs_t.shape[:2])} "
                        f"and rewards shape={tuple(rew_t.shape)}."
                    )
        T, B = rT, rB
    else:
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
                if o0 == N:
                    T, B = o0, 1
                else:
                    raise ValueError(
                        "compute_gae: cannot infer (T,B) from 1-D rewards/dones and obs shape. "
                        f"Got N={N}, obs leading dims={tuple(obs_t.shape[:2])}."
                    )
        else:
            o0 = int(obs_t.size(0))
            if o0 != N:
                raise ValueError(
                    "compute_gae: rewards/dones length does not match obs time length. "
                    f"N={N}, T(obs)={o0}."
                )
            T, B = o0, 1
        rew_t = rew_t.reshape(T, B)
        done_t = done_t.reshape(T, B)

    if obs_t.dim() >= 2 and (int(obs_t.size(0)) == B and int(obs_t.size(1)) == T):
        obs_t = obs_t.transpose(0, 1)
        if next_obs_t is not None:
            next_obs_t = next_obs_t.transpose(0, 1)

    if obs_t.dim() >= 2:
        if not (int(obs_t.size(0)) == T and (B == 1 or int(obs_t.size(1)) == B)):
            raise ValueError(
                "compute_gae: observation tensor is not time-major (T,B,...). "
                f"Expected leading dims (T,B)=({T},{B}); got {tuple(obs_t.shape[:2])}."
            )

    return obs_t, next_obs_t, rew_t, done_t, T, B


def _coerce_mask_to_TB(mask: Tensor, T: int, B: int, like_done: Tensor) -> Tensor:
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
) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping/dict-like object")

    device = next(value_fn.parameters()).device

    obs = _pick(batch, "obs", "observations")
    next_obs = _pick(batch, "next_obs", "next_observations")
    rewards = _pick(batch, "rewards", "r_total", "r")
    dones_any = _pick(batch, "dones", "done", "terminals")
    terminals_raw = _pick(batch, "terminals", "terminal", "terms")
    trunc_raw = _pick(batch, "truncations", "timeouts", "truncated", "time_limits")

    if obs is None or rewards is None:
        raise KeyError("batch must contain observations and rewards.")
    if dones_any is None and terminals_raw is None and trunc_raw is None:
        raise KeyError("batch must contain 'dones' or separate 'terminals'/'truncations' masks.")

    obs_t = _to_tensor(obs, device)
    next_obs_t = None if next_obs is None else _to_tensor(next_obs, device)
    rew_t = _to_tensor(rewards, device, dtype=torch.float32)

    done_base = torch.zeros_like(rew_t, dtype=torch.float32)
    if dones_any is not None:
        done_base = _to_tensor(dones_any, device, dtype=torch.float32)

    obs_t, next_obs_t, rew_t, done_t, T, B = _ensure_time_batch_layout(
        obs_t, next_obs_t, rew_t, done_base
    )

    term_t: Tensor | None = None
    trunc_t: Tensor | None = None
    if terminals_raw is not None:
        term_t = _coerce_mask_to_TB(_to_tensor(terminals_raw, device), T, B, done_t)
    if trunc_raw is not None:
        trunc_t = _coerce_mask_to_TB(_to_tensor(trunc_raw, device), T, B, done_t)

    has_term = term_t is not None
    has_trunc = trunc_t is not None

    if not has_term and not has_trunc:
        done_rec = done_t
        term = None
        trunc = None
    else:
        trunc = trunc_t if trunc_t is not None else torch.zeros_like(done_t)
        if term_t is not None:
            term = term_t
        else:
            term = torch.zeros_like(done_t)
            if dones_any is not None and trunc_t is not None:
                # If callers provide `dones` + `truncations` but omit `terminals`, infer terminals as
                # done - trunc to preserve episode boundaries without leaking across timeouts.
                term = torch.clamp(done_t - trunc, min=0.0, max=1.0)
        done_rec = torch.clamp(term + trunc, max=1.0)

    with torch.no_grad():
        v_t = value_fn(obs_t).view(T, B)

        if next_obs_t is not None:
            v_tp1 = value_fn(next_obs_t).view(T, B)
        else:
            v_tp1 = torch.zeros_like(v_t)
            if T > 1:
                v_tp1[:-1] = v_t[1:]
            v_tp1[-1] = 0.0

        rew_eff = rew_t
        if bootstrap_on_timeouts and trunc_t is not None and trunc is not None:
            term_eff = term if term is not None else torch.zeros_like(done_t)
            trunc_boot = trunc * (1.0 - term_eff)
            rew_eff = rew_t + float(gamma) * trunc_boot * v_tp1

        adv = torch.zeros_like(v_t)
        last_adv = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            delta = rew_eff[t] + gamma * (1.0 - done_rec[t]) * v_tp1[t] - v_t[t]
            last_adv = delta + gamma * lam * (1.0 - done_rec[t]) * last_adv
            adv[t] = last_adv

        v_targets = adv + v_t

    return adv.reshape(-1), v_targets.reshape(-1)
