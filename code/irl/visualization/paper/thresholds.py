from __future__ import annotations

from typing import Any

import numpy as np

_SUPPORTED_SCORE_ENVS: set[str] = {
    "Ant-v5",
    "BipedalWalker-v3",
    "CarRacing-v3",
    "HalfCheetah-v5",
    "Humanoid-v5",
    "MountainCar-v0",
}

_KNOWN_SCORE_THRESHOLDS: dict[str, float] = {
    "Ant-v5": 6000.0,
    "BipedalWalker-v3": 300.0,
    "CarRacing-v3": 900.0,
    "HalfCheetah-v5": 4800.0,
    "Humanoid-v5": 6000.0,
    "MountainCar-v0": -110.0,
}

SOLVED_THRESHOLD_LABEL: str = "Solved threshold"

_SOLVED_THRESHOLD_COLOR: str = "black"
_SOLVED_THRESHOLD_ALPHA: float = 0.35
_SOLVED_THRESHOLD_LINEWIDTH: float = 1.0
_SOLVED_THRESHOLD_LINESTYLE: str = "--"


def solved_threshold_legend_handle(plt: Any) -> object:
    return plt.Line2D(
        [],
        [],
        color=_SOLVED_THRESHOLD_COLOR,
        linewidth=float(_SOLVED_THRESHOLD_LINEWIDTH),
        linestyle=_SOLVED_THRESHOLD_LINESTYLE,
        alpha=float(_SOLVED_THRESHOLD_ALPHA),
    )


def reward_threshold_from_gym_spec(env_id: str) -> float | None:
    try:
        import gymnasium as gym  # type: ignore
    except Exception:
        return None

    try:
        spec = gym.spec(str(env_id))
    except Exception:
        return None

    rt = getattr(spec, "reward_threshold", None)
    if rt is None:
        return None

    try:
        v = float(rt)
    except Exception:
        return None

    return v if np.isfinite(v) else None


def reward_threshold_from_known(env_id: str) -> float | None:
    key = str(env_id).strip()
    if key in _KNOWN_SCORE_THRESHOLDS:
        return float(_KNOWN_SCORE_THRESHOLDS[key])
    return None


def solved_threshold(env_id: str) -> float | None:
    rt = reward_threshold_from_gym_spec(str(env_id))
    if rt is not None and np.isfinite(float(rt)):
        return float(rt)

    rt = reward_threshold_from_known(str(env_id))
    if rt is not None and np.isfinite(float(rt)):
        return float(rt)

    return None


def add_solved_threshold_line(ax, env_id: str) -> float | None:
    thr = solved_threshold(str(env_id))
    if thr is None:
        return None

    thr_f = float(thr)
    if not np.isfinite(thr_f):
        return None

    try:
        ax.axhline(
            thr_f,
            linewidth=float(_SOLVED_THRESHOLD_LINEWIDTH),
            alpha=float(_SOLVED_THRESHOLD_ALPHA),
            color=_SOLVED_THRESHOLD_COLOR,
            linestyle=_SOLVED_THRESHOLD_LINESTYLE,
            label=SOLVED_THRESHOLD_LABEL,
            zorder=1,
        )
    except Exception:
        return None

    return thr_f


def fmt_threshold(v: float) -> str:
    if not np.isfinite(float(v)):
        return "nan"
    if abs(float(v) - round(float(v))) < 1e-9:
        return str(int(round(float(v))))
    return f"{float(v):.3g}"
