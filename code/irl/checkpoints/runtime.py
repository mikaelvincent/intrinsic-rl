from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from typing import Any

from irl.paper_defaults import DEFAULT_EVAL_EPISODES
from irl.pipelines.runtime import build_obs_normalizer as _build_obs_normalizer
from irl.pipelines.runtime import extract_env_runtime as _extract_env_runtime


@dataclass(frozen=True)
class EvalSettings:
    interval_steps: int
    episodes: int
    device: str


def _normalize_eval_device(raw: Any) -> str:
    if raw is None:
        return "cpu"

    dev = str(raw).strip() or "cpu"
    low = dev.lower()

    if low.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"

    return dev


def extract_eval_settings(payload: Any) -> EvalSettings:
    cfg = payload.get("cfg") if isinstance(payload, ABCMapping) else None
    if not isinstance(cfg, ABCMapping):
        cfg = {}

    ev = cfg.get("evaluation") if isinstance(cfg, ABCMapping) else None
    if not isinstance(ev, ABCMapping):
        ev = {}

    raw_interval = ev.get("interval_steps", None)
    try:
        interval_steps = int(raw_interval)
    except Exception:
        interval_steps = 0
    interval_steps = max(0, int(interval_steps))

    raw_episodes = ev.get("episodes", None)
    episodes = int(DEFAULT_EVAL_EPISODES)
    try:
        ep_i = int(raw_episodes)
        if ep_i > 0:
            episodes = int(ep_i)
    except Exception:
        episodes = int(DEFAULT_EVAL_EPISODES)

    raw_device = cfg.get("device", None) if isinstance(cfg, ABCMapping) else None
    device = _normalize_eval_device(raw_device)

    return EvalSettings(
        interval_steps=int(interval_steps),
        episodes=int(episodes),
        device=str(device),
    )


def extract_env_settings(cfg_mapping: Any) -> dict[str, object]:
    return _extract_env_runtime(cfg_mapping)


def build_obs_normalizer(payload: Any):
    return _build_obs_normalizer(payload)
