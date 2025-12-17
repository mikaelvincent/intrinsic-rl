from __future__ import annotations

from typing import Any

import typer

QUICK_EPISODES = 5


def validate_policy_mode(
    policy: Any,
    *,
    allowed: tuple[str, ...] = ("mode", "sample"),
    option: str = "--policy",
) -> str:
    pm = str(policy).strip().lower()
    allowed_norm = {str(a).strip().lower() for a in allowed}
    if pm not in allowed_norm:
        allowed_s = ", ".join(str(a).strip().lower() for a in allowed)
        raise typer.BadParameter(f"{option} must be one of: {allowed_s}")
    return pm


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


def _cfg_total_steps(cfg: Any) -> int | None:
    try:
        exp = getattr(cfg, "exp", None)
        ts = getattr(exp, "total_steps", None) if exp is not None else None
        if ts is None:
            return None
        return int(ts)
    except Exception:
        return None


def resolve_total_steps(
    cfg: Any,
    cli_total_steps: int | None,
    *,
    default_total_steps: int = 10_000,
    prefer_cfg: bool = False,
) -> int:
    cfg_steps = _cfg_total_steps(cfg)
    if bool(prefer_cfg) and cfg_steps is not None:
        return int(cfg_steps)
    if cli_total_steps is not None:
        return int(cli_total_steps)
    if cfg_steps is not None:
        return int(cfg_steps)
    return int(default_total_steps)
