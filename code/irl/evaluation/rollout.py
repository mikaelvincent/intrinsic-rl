from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from irl.cli.validators import normalize_policy_mode
from irl.intrinsic.factory import compute_intrinsic_batch as _compute_intrinsic_batch
from irl.pipelines.policy_rollout import iter_policy_rollout
from irl.utils.seeding import seed_torch_only as _seed_torch_only

NormalizeFn = Callable[[np.ndarray], np.ndarray]


def _seed_torch(seed: int) -> None:
    _seed_torch_only(seed)


def glpe_gate_and_intrinsic_no_update(
    mod: object, obs_1d: torch.Tensor, next_obs_1d: torch.Tensor
) -> tuple[int, float] | None:
    if not (hasattr(mod, "icm") and hasattr(mod, "store")):
        return None
    stats = getattr(mod, "_stats", None)
    if not isinstance(stats, dict):
        return None
    try:
        with torch.no_grad():
            b_obs = obs_1d.unsqueeze(0)
            b_next = next_obs_1d.unsqueeze(0)

            phi_t = mod.icm._phi(b_obs)
            rid = int(mod.store.locate(phi_t.detach().cpu().numpy().reshape(-1)))

            st = stats.get(rid)
            gate = 1 if st is None else int(getattr(st, "gate", 1))

            lp_raw = 0.0
            if st is not None and int(getattr(st, "count", 0)) > 0:
                lp_raw = max(
                    0.0,
                    float(getattr(st, "ema_long", 0.0) - getattr(st, "ema_short", 0.0)),
                )

            impact_raw_t = mod._impact_per_sample(b_obs, b_next)
            impact_raw = float(impact_raw_t.view(-1)[0].item())

            a_imp = float(getattr(mod, "alpha_impact", 1.0))
            a_lp = float(getattr(mod, "alpha_lp", 0.0))

            if bool(getattr(mod, "_normalize_inside", False)) and hasattr(mod, "_rms"):
                imp_n, lp_n = mod._rms.normalize(
                    np.asarray([impact_raw], dtype=np.float32),
                    np.asarray([lp_raw], dtype=np.float32),
                )
                combined = a_imp * float(imp_n[0]) + a_lp * float(lp_n[0])
            else:
                combined = a_imp * float(impact_raw) + a_lp * float(lp_raw)

            return int(gate), float(gate) * float(combined)
    except Exception:
        return None


@dataclass(frozen=True)
class Trajectory:
    obs: list[np.ndarray]
    rewards_ext: list[float]
    gates: list[int]
    intrinsic: list[float]
    gate_source: str | None
    intrinsic_semantics: str | None


@dataclass(frozen=True)
class RolloutResult:
    returns: list[float]
    lengths: list[int]
    trajectory: Trajectory | None


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
    is_intrinsic_method = is_glpe_family or method_l in {"icm", "rnd", "ride", "riac"}
    want_traj = bool(save_traj) and not bool(is_image)

    traj_obs: list[np.ndarray] = []
    traj_rewards_ext: list[float] = []
    traj_gates: list[int] = []
    traj_int_vals: list[float] = []
    traj_gate_source: str | None = None
    traj_intrinsic_semantics: str | None = None

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
                traj_rewards_ext.append(float(step_rec.reward))

                gate_val = 1
                int_val = 0.0

                if is_glpe_family:
                    gate_source = "recomputed" if intrinsic_module is not None else "missing_intrinsic"
                else:
                    gate_source = "n/a"

                if intrinsic_module is None:
                    intrinsic_semantics = "missing_intrinsic" if is_intrinsic_method else "none"
                else:
                    if is_glpe_family:
                        intrinsic_semantics = "frozen_checkpoint"
                    elif method_l == "ride":
                        intrinsic_semantics = "unbinned_impact"
                    else:
                        intrinsic_semantics = "compute_batch"

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
                                intrinsic_semantics = "frozen_checkpoint"
                            else:
                                intrinsic_semantics = "unavailable"
                        else:
                            r_out = _compute_intrinsic_batch(
                                intrinsic_module,
                                method_l,
                                step_rec.obs_t,
                                next_t,
                                step_rec.act_t,
                            )
                            int_val = float(r_out.view(-1)[0].item())
                    except Exception:
                        intrinsic_semantics = "unavailable"

                if bool(getattr(step_rec, "terminated", False)):
                    int_val = 0.0

                traj_gates.append(int(gate_val))
                traj_int_vals.append(float(int_val))

                if traj_gate_source is None:
                    traj_gate_source = gate_source
                elif traj_gate_source != gate_source:
                    traj_gate_source = "mixed"

                if traj_intrinsic_semantics is None:
                    traj_intrinsic_semantics = intrinsic_semantics
                elif traj_intrinsic_semantics != intrinsic_semantics:
                    traj_intrinsic_semantics = "mixed"

        returns.append(float(ep_ret))
        lengths.append(int(ep_len))

    traj = (
        Trajectory(
            obs=traj_obs,
            rewards_ext=traj_rewards_ext,
            gates=traj_gates,
            intrinsic=traj_int_vals,
            gate_source=traj_gate_source,
            intrinsic_semantics=traj_intrinsic_semantics,
        )
        if want_traj and traj_obs
        else None
    )
    return RolloutResult(returns=returns, lengths=lengths, trajectory=traj)
