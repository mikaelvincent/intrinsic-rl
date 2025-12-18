from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from irl.cli.validators import normalize_policy_mode
from irl.pipelines.policy_rollout import iter_policy_rollout

from .glpe import glpe_gate_and_intrinsic_no_update

NormalizeFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Trajectory:
    obs: list[np.ndarray]
    gates: list[int]
    intrinsic: list[float]
    gate_source: str | None


@dataclass(frozen=True)
class RolloutResult:
    returns: list[float]
    lengths: list[int]
    trajectory: Trajectory | None


def _seed_torch(seed: int) -> None:
    s = int(seed)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(s)
        except Exception:
            pass


def run_eval_episodes(
    *,
    env: Any,
    policy: Any,
    act_space: Any,
    device: torch.device,
    policy_mode: str,
    episode_seeds: Sequence[int],
    normalize_obs: NormalizeFn,
    save_traj: bool,
    is_image: bool,
    intrinsic_module: Any | None,
    method: str,
) -> RolloutResult:
    mode = normalize_policy_mode(policy_mode, allowed=("mode", "sample"), name="policy_mode")

    returns: list[float] = []
    lengths: list[int] = []

    method_l = str(method).strip().lower()
    is_glpe_family = method_l.startswith("glpe")
    want_traj = bool(save_traj) and not bool(is_image)

    traj_obs: list[np.ndarray] = []
    traj_gates: list[int] = []
    traj_int_vals: list[float] = []
    traj_gate_source: str | None = None

    for ep_seed in episode_seeds:
        _seed_torch(int(ep_seed))
        obs0, _ = env.reset(seed=int(ep_seed))

        ep_ret = 0.0
        ep_len = 0

        for step_rec in iter_policy_rollout(
            env=env,
            policy=policy,
            obs0=obs0,
            act_space=act_space,
            device=device,
            policy_mode=mode,
            normalize_obs=normalize_obs,
            max_steps=None,
        ):
            ep_ret += float(step_rec.reward)
            ep_len += 1

            if want_traj:
                traj_obs.append(step_rec.obs_raw.copy())

                gate_val = 1
                int_val = 0.0
                gate_source = "recomputed" if is_glpe_family else "n/a"

                if intrinsic_module is not None:
                    try:
                        next_t = torch.as_tensor(
                            normalize_obs(step_rec.next_obs_raw),
                            dtype=torch.float32,
                            device=device,
                        )

                        if is_glpe_family:
                            res = glpe_gate_and_intrinsic_no_update(
                                intrinsic_module, step_rec.obs_t, next_t
                            )
                            if res is not None:
                                gate_val, int_val = res
                                gate_source = "checkpoint"
                        else:
                            try:
                                r_out = intrinsic_module.compute_batch(
                                    step_rec.obs_t.unsqueeze(0),
                                    next_t.unsqueeze(0),
                                    step_rec.act_t.unsqueeze(0),
                                    reduction="none",
                                )
                                int_val = float(r_out.mean().item())
                            except Exception:
                                pass
                    except Exception:
                        pass

                traj_gates.append(int(gate_val))
                traj_int_vals.append(float(int_val))

                if traj_gate_source is None:
                    traj_gate_source = gate_source
                elif traj_gate_source != gate_source:
                    traj_gate_source = "mixed"

        returns.append(float(ep_ret))
        lengths.append(int(ep_len))

    traj = (
        Trajectory(
            obs=traj_obs,
            gates=traj_gates,
            intrinsic=traj_int_vals,
            gate_source=traj_gate_source,
        )
        if want_traj and traj_obs
        else None
    )
    return RolloutResult(returns=returns, lengths=lengths, trajectory=traj)
