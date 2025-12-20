from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch

from irl.runtime.envvars import ensure_mujoco_gl as _ensure_mujoco_gl
from irl.utils.loggers import get_logger

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


def ensure_mujoco_gl(env_id: str) -> str:
    return _ensure_mujoco_gl(env_id)
