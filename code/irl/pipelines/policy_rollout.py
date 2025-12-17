from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
import torch

from irl.cli.validators import normalize_policy_mode

NormalizeFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PolicyRolloutStep:
    obs_raw: np.ndarray
    obs_in: np.ndarray
    obs_t: torch.Tensor
    act_t: torch.Tensor
    action_env: Any
    reward: float
    terminated: bool
    truncated: bool
    next_obs_raw: np.ndarray


def _as_np_obs(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=np.float32)


def iter_policy_rollout(
    *,
    env: Any,
    policy: Any,
    obs0: Any,
    act_space: Any,
    device: torch.device,
    policy_mode: str,
    normalize_obs: NormalizeFn | None = None,
    max_steps: int | None = None,
) -> Iterator[PolicyRolloutStep]:
    mode = normalize_policy_mode(policy_mode, allowed=("mode", "sample"), name="policy_mode")

    if max_steps is not None and int(max_steps) < 0:
        raise ValueError("max_steps must be >= 0 or None")

    obs = obs0
    steps = 0

    while True:
        if max_steps is not None and steps >= int(max_steps):
            return

        obs_raw = _as_np_obs(obs)
        obs_in = normalize_obs(obs_raw) if normalize_obs is not None else obs_raw

        with torch.no_grad():
            obs_t = torch.as_tensor(obs_in, dtype=torch.float32, device=device)
            dist = policy.distribution(obs_t)
            act_t = dist.mode() if mode == "mode" else dist.sample()
            a_np = act_t.detach().cpu().numpy()

        if hasattr(act_space, "n"):
            action_env: Any = int(a_np.item())
        else:
            action_env = a_np.reshape(-1)

        next_obs, r, term, trunc, _info = env.step(action_env)
        next_obs_raw = _as_np_obs(next_obs)

        yield PolicyRolloutStep(
            obs_raw=obs_raw,
            obs_in=obs_in,
            obs_t=obs_t,
            act_t=act_t,
            action_env=action_env,
            reward=float(r),
            terminated=bool(term),
            truncated=bool(trunc),
            next_obs_raw=next_obs_raw,
        )

        obs = next_obs
        steps += 1

        if bool(term) or bool(trunc):
            return
