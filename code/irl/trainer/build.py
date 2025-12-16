from __future__ import annotations

import os
import sys
from ctypes import CDLL
from ctypes.util import find_library
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch

from irl.utils.loggers import get_logger, log_mujoco_gl_default, log_mujoco_gl_preserve

_LOG = get_logger(__name__)


def single_spaces(env) -> Tuple:
    obs_space = getattr(env, "single_observation_space", None) or env.observation_space
    act_space = getattr(env, "single_action_space", None) or env.action_space
    return obs_space, act_space


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def default_run_dir(cfg) -> Path:
    base = Path("runs")
    env_id = cfg.env.id.replace("/", "-")
    tag = f"{cfg.method}__{env_id}__seed{cfg.seed}__{now_tag()}"
    return base / tag


def ensure_device(dev_str: str) -> torch.device:
    d = dev_str.strip().lower()
    if d.startswith("cuda") and not torch.cuda.is_available():
        _LOG.warning(
            "CUDA requested via device=%r but no CUDA device is available; falling back to CPU.",
            dev_str,
        )
        return torch.device("cpu")
    return torch.device(dev_str)


_MUJOCO_ENV_HINTS = (
    "Ant",
    "HalfCheetah",
    "Humanoid",
    "Hopper",
    "Walker2d",
    "Swimmer",
    "Reacher",
    "Pusher",
    "InvertedPendulum",
    "InvertedDoublePendulum",
)


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

    _LOG.warning(
        "MUJOCO_GL not set; EGL/OSMesa libraries not found. Rendering may fail unless a GL backend is installed."
    )
    return ""
