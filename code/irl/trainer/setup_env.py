from __future__ import annotations

from typing import Any, Optional, Tuple

from irl.cfg import Config
from irl.envs.builder import make_env
from irl.utils.loggers import log_domain_randomization, log_intrinsic_norm_hint


def _build_env(cfg: Config, *, logger) -> Any:
    async_vector = bool(getattr(cfg.env, "async_vector", False))
    env = make_env(
        env_id=cfg.env.id,
        num_envs=cfg.env.vec_envs,
        seed=cfg.seed,
        frame_skip=cfg.env.frame_skip,
        domain_randomization=cfg.env.domain_randomization,
        discrete_actions=cfg.env.discrete_actions,
        car_action_set=cfg.env.car_discrete_action_set,
        render_mode=None,
        async_vector=async_vector,
        make_kwargs=None,
    )
    if int(cfg.env.vec_envs) > 1:
        logger.info(
            "Vector env mode: %s (num_envs=%d) for env_id=%s",
            "Async" if async_vector else "Sync",
            int(cfg.env.vec_envs),
            cfg.env.id,
        )
    return env


def _reset_env(env: Any, seed: int | None) -> Tuple[Any, Any]:
    if seed is None:
        return env.reset()

    B = int(getattr(env, "num_envs", 1))
    if B <= 1:
        try:
            return env.reset(seed=int(seed))
        except (TypeError, ValueError):
            return env.reset()

    seeds = [int(seed) + i for i in range(B)]
    try:
        return env.reset(seed=seeds)
    except (TypeError, ValueError):
        try:
            return env.reset(seed=int(seed))
        except (TypeError, ValueError):
            return env.reset()


def _log_reset_diagnostics(
    *,
    env: Any,
    intrinsic_module: Optional[Any],
    method_l: str,
    intrinsic_outputs_normalized_flag: Optional[bool],
    seed: int | None,
) -> Any:
    printed_dr_hint = False
    printed_intr_norm_hint = False

    obs, info = _reset_env(env, seed)

    try:
        if isinstance(info, dict) and ("dr_applied" in info) and not printed_dr_hint:
            diag = info.get("dr_applied")
            msg = ""
            if isinstance(diag, dict):
                mj = int(diag.get("mujoco", 0))
                b2 = int(diag.get("box2d", 0))
                msg = f"mujoco={mj}, box2d={b2}"
            elif isinstance(diag, (list, tuple)):
                mj = 0
                b2 = 0
                n = 0
                for d in diag:
                    if isinstance(d, dict):
                        mj += int(d.get("mujoco", 0))
                        b2 += int(d.get("box2d", 0))
                        n += 1
                msg = f"mujoco={mj}, box2d={b2} (across {n} envs)"
            else:
                msg = str(diag)
            log_domain_randomization(msg)
            printed_dr_hint = True
    except Exception:
        pass

    if intrinsic_module is not None and not printed_intr_norm_hint:
        try:
            outputs_norm_flag = (
                intrinsic_outputs_normalized_flag
                if intrinsic_outputs_normalized_flag is not None
                else bool(getattr(intrinsic_module, "outputs_normalized", False))
            )
            log_intrinsic_norm_hint(method_l, bool(outputs_norm_flag))
            printed_intr_norm_hint = True
        except Exception:
            pass

    return obs
