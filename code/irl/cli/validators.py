from __future__ import annotations

from typing import Any


def normalize_policy_mode(
    policy: Any,
    *,
    allowed: tuple[str, ...] = ("mode", "sample"),
    name: str = "policy_mode",
) -> str:
    pm = str(policy).strip().lower()
    allowed_norm = tuple(str(a).strip().lower() for a in allowed)
    if pm not in set(allowed_norm):
        allowed_s = ", ".join(allowed_norm)
        raise ValueError(f"{name} must be one of: {allowed_s}")
    return pm
