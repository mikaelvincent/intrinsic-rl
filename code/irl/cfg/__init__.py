"""Configuration package (skeleton).

Actual YAML->dataclass loaders and schema live here in Step 3 of Sprint 0. For now we expose a minimal placeholder API
so other modules may import without failing.
"""

from __future__ import annotations

from typing import Mapping, Any


ConfigLike = Mapping[str, Any]
"""A loose mapping type used until the concrete dataclasses are added."""


def load_config(path: str) -> ConfigLike:  # pragma: no cover - to be implemented later
    """Placeholder for the real config loader.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        A mapping-like object representing the configuration.

    Raises:
        NotImplementedError: Always, until Sprint 0 Step 3.
    """
    raise NotImplementedError("Config loader will be implemented in Sprint 0 — Step 3.")
