from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

_STYLE_RCPARAMS: dict[str, Any] = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
}


def apply_rcparams_paper():
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    plt.rcParams.update(_STYLE_RCPARAMS)
    return plt


def save_fig_atomic(
    fig,
    path: Path,
    *,
    dpi: int = 150,
    bbox_inches: str = "tight",
    format: str | None = None,
) -> None:
    from irl.utils.checkpoint import atomic_replace

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    fmt = format
    if fmt is None:
        fmt = path.suffix.lstrip(".").lower() or "png"
    else:
        fmt = str(fmt).strip().lower() or (path.suffix.lstrip(".").lower() or "png")

    fig.savefig(str(tmp), dpi=int(dpi), bbox_inches=bbox_inches, format=fmt)
    atomic_replace(tmp, path)


_ENV_ORDER: tuple[str, ...] = (
    "MountainCar-v0",
    "BipedalWalker-v3",
    "HalfCheetah-v5",
    "Ant-v5",
    "CarRacing-v3",
    "Humanoid-v5",
)

_ENV_RANK: dict[str, int] = {env: i for i, env in enumerate(_ENV_ORDER)}


def env_sort_key(env_id: object) -> tuple[int, str]:
    s = str(env_id).strip()
    if not s:
        return (10_000, "")
    key = s.replace("/", "-")
    rank = _ENV_RANK.get(key)
    if rank is None:
        rank = 10_000
    return (int(rank), key.lower())


def sort_env_ids(env_ids: Iterable[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for e in env_ids:
        s = str(e).strip()
        if not s or s in seen:
            continue
        out.append(s)
        seen.add(s)
    out.sort(key=env_sort_key)
    return out
