from __future__ import annotations

from typing import Any

import typer

from irl.utils.steps import resolve_total_steps as _resolve_total_steps

from .validators import normalize_policy_mode

QUICK_EPISODES = 5


def validate_policy_mode(
    policy: Any,
    *,
    allowed: tuple[str, ...] = ("mode", "sample"),
    option: str = "--policy",
) -> str:
    try:
        return normalize_policy_mode(policy, allowed=allowed, name=option)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def resolve_default_method_for_entrypoint(
    *,
    config_provided: bool,
    method: str | None,
    default_no_config: str,
) -> str | None:
    if method is not None:
        return str(method)
    if not bool(config_provided):
        return str(default_no_config)
    return None


def resolve_total_steps(
    cfg: Any,
    cli_total_steps: int | None,
    *,
    default_total_steps: int = 10_000,
    prefer_cfg: bool = False,
) -> int:
    return _resolve_total_steps(
        cfg,
        cli_total_steps,
        default_total_steps=int(default_total_steps),
        prefer_cfg=bool(prefer_cfg),
        align_to_vec_envs=False,
    )
