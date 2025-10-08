"""Advantage estimation (skeleton)."""

from __future__ import annotations

from typing import Any, Tuple


def compute_gae(
    batch: Any, value_fn: Any, gamma: float, lam: float
) -> Tuple[Any, Any]:  # pragma: no cover - placeholder
    """Placeholder for GAE advantage and value-target computation.

    Returns:
        (advantages, value_targets)
    """
    raise NotImplementedError(
        "GAE computation will be implemented in Sprint 0 — Step 5."
    )
