"""Checkpoint utilities facade.

This module keeps the original public API stable while delegating the
implementation to smaller, focused modules:

* :mod:`irl.utils.atomic_files`  — atomic text/bytes helpers.
* :mod:`irl.utils.config_hash`  — deterministic configuration hashing.
* :mod:`irl.utils.checkpoint_manager` — CheckpointManager and torch I/O.

Existing imports such as::

    from irl.utils.checkpoint import (
        atomic_replace,
        atomic_write_text,
        atomic_write_bytes,
        compute_cfg_hash,
        CheckpointManager,
        load_checkpoint,
    )

continue to work unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .atomic_files import atomic_replace, atomic_write_text, atomic_write_bytes
from .config_hash import compute_cfg_hash
from .checkpoint_manager import CheckpointManager, load_checkpoint

__all__ = [
    "atomic_replace",
    "atomic_write_text",
    "atomic_write_bytes",
    "compute_cfg_hash",
    "CheckpointManager",
    "load_checkpoint",
]


# Re-export types for type-checkers and callers that used this module
# as the single checkpoint/atomic I/O surface.
def _reexport_sentinel(*_args: Any, **_kwargs: Any) -> None:
    """No-op helper to make it obvious this module is a thin facade.

    This function has no runtime role; it simply documents that the
    implementation lives in sibling modules.
    """
    return None
