from __future__ import annotations

import os
import sys
from ctypes import CDLL
from ctypes.util import find_library
from typing import Tuple

from irl.utils.loggers import get_logger, log_mujoco_gl_default, log_mujoco_gl_preserve

_MUJOCO_ENV_HINTS: tuple[str, ...] = (
    "Ant",
    "HalfCheetah",
    "Humanoid",
)


def disable_nnpack() -> None:
    os.environ["ATEN_NNPACK_ENABLED"] = "0"


def ensure_sdl_dummy() -> None:
    if "SDL_VIDEODRIVER" not in os.environ:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


def ensure_mujoco_gl(env_id: str) -> str:
    is_mujoco = any(hint in str(env_id) for hint in _MUJOCO_ENV_HINTS)
    if not is_mujoco:
        return os.environ.get("MUJOCO_GL", "") or ""

    current = os.environ.get("MUJOCO_GL")
    if current:
        if sys.platform.startswith("linux"):
            log_mujoco_gl_preserve(current)
        return current

    if not sys.platform.startswith("linux"):
        return ""

    def _can_load(names: Tuple[str, ...]) -> bool:
        for name in names:
            lib_path = find_library(name)
            if not lib_path:
                continue
            try:
                CDLL(lib_path)
                return True
            except OSError:
                continue
        return False

    for backend, libs in (("egl", ("EGL",)), ("osmesa", ("OSMesa", "osmesa"))):
        if _can_load(libs):
            os.environ["MUJOCO_GL"] = backend
            log_mujoco_gl_default(backend)
            return backend

    get_logger(__name__).warning(
        "MUJOCO_GL not set; EGL/OSMesa libraries not found. Rendering may fail unless a GL backend is installed."
    )
    return ""
