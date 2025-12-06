from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Tuple
import os
import sys

import torch

from irl.utils.loggers import (
    get_logger,
    log_mujoco_gl_default,
    log_mujoco_gl_preserve,
)

_LOG = get_logger(__name__)


def single_spaces(env) -> Tuple:
    """Return (obs_space, action_space) for both single and vector envs."""
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
    """Resolve a torch.device, warning once if CUDA is requested but unavailable."""
    d = dev_str.strip().lower()
    if d.startswith("cuda") and not torch.cuda.is_available():
        _LOG.warning(
            "CUDA requested via device=%r but no CUDA device is available; falling back to CPU.",
            dev_str,
        )
        return torch.device("cpu")
    return torch.device(dev_str)


# -------------------- MuJoCo headless rendering helper --------------------


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
    """Ensure MUJOCO_GL is sensible for MuJoCo tasks.

    Behavior:
    * If `env_id` looks like a MuJoCo task and MUJOCO_GL is *unset*:
        - On Linux: set MUJOCO_GL='egl' (common headless default).
        - On Windows/macOS: leave MUJOCO_GL unset and stay quiet (hint is Linux-only).
    * If MUJOCO_GL is already set:
        - On Linux: emit an informational log once per process.
        - On Windows/macOS: return the value without additional noise.

    Returns
    -------
    str
        The value of MUJOCO_GL after this call ('' if unset).
    """
    is_mujoco = any(hint in str(env_id) for hint in _MUJOCO_ENV_HINTS)
    if not is_mujoco:
        return os.environ.get("MUJOCO_GL", "") or ""

    current = os.environ.get("MUJOCO_GL")
    if current:
        # Only log on Linux to avoid noisy hints on Windows/macOS.
        if sys.platform.startswith("linux"):
            log_mujoco_gl_preserve(current)
        return current

    # If unset, choose a safe default on Linux; stay silent elsewhere.
    if not sys.platform.startswith("linux"):
        # MUJOCO_GL is primarily relevant for headless Linux; on other platforms
        # a missing variable is usually fine and we avoid emitting extra logs.
        return ""

    os.environ["MUJOCO_GL"] = "egl"
    log_mujoco_gl_default("egl")
    return "egl"
