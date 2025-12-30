from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from irl.visualization.style import DPI, LEGEND_FONTSIZE

_STYLE_RCPARAMS: dict[str, Any] = {
    "figure.dpi": int(DPI),
    "savefig.dpi": int(DPI),
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": int(LEGEND_FONTSIZE),
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _disable_matplotlib_titles() -> None:
    import matplotlib.axes
    import matplotlib.figure

    if bool(getattr(matplotlib.axes.Axes, "_irl_titles_disabled", False)):
        return

    orig_set_title = matplotlib.axes.Axes.set_title
    orig_suptitle = matplotlib.figure.Figure.suptitle

    def _set_title_blank(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        txt = orig_set_title(self, "", pad=0.0)
        try:
            txt.set_visible(False)
        except Exception:
            pass
        return txt

    def _suptitle_blank(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        txt = orig_suptitle(self, "", y=1.0)
        try:
            txt.set_visible(False)
        except Exception:
            pass
        return txt

    matplotlib.axes.Axes.set_title = _set_title_blank  # type: ignore[assignment]
    matplotlib.figure.Figure.suptitle = _suptitle_blank  # type: ignore[assignment]
    setattr(matplotlib.axes.Axes, "_irl_titles_disabled", True)


def apply_rcparams_paper():
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    plt.rcParams.update(_STYLE_RCPARAMS)
    _disable_matplotlib_titles()
    return plt


def save_fig_atomic(
    fig,
    path: Path,
    *,
    dpi: int = int(DPI),
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
