from __future__ import annotations

from typing import Any


def _get_cfg_total_steps(cfg: Any) -> int | None:
    try:
        exp = getattr(cfg, "exp", None)
        ts = getattr(exp, "total_steps", None) if exp is not None else None
        return None if ts is None else int(ts)
    except Exception:
        return None


def _get_vec_envs(cfg: Any) -> int:
    try:
        env = getattr(cfg, "env", None)
        v = getattr(env, "vec_envs", 1) if env is not None else 1
        return max(1, int(v))
    except Exception:
        return 1


def resolve_total_steps(
    cfg: Any,
    requested_steps: int | None,
    *,
    default_total_steps: int = 10_000,
    prefer_cfg: bool = False,
    align_to_vec_envs: bool = True,
) -> int:
    cfg_steps = _get_cfg_total_steps(cfg)

    if bool(prefer_cfg) and cfg_steps is not None:
        steps = int(cfg_steps)
    elif requested_steps is not None:
        steps = int(requested_steps)
    elif cfg_steps is not None:
        steps = int(cfg_steps)
    else:
        steps = int(default_total_steps)

    if bool(align_to_vec_envs):
        vec_envs = _get_vec_envs(cfg)
        if int(vec_envs) > 1:
            steps = (int(steps) // int(vec_envs)) * int(vec_envs)

    return int(steps)
