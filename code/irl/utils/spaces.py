from __future__ import annotations

from typing import Any


def is_image_space(space: Any) -> bool:
    shape = getattr(space, "shape", None)
    if shape is None:
        return False
    try:
        return len(shape) >= 2
    except Exception:
        return False
