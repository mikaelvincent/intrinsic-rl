from __future__ import annotations

from pathlib import Path
from typing import Any


_PAPER_RCPARAMS: dict[str, Any] = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}


def apply_rcparams_paper():
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    plt.rcParams.update(_PAPER_RCPARAMS)
    return plt


def save_fig_atomic(
    fig,
    path: Path,
    *,
    dpi: int = 300,
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
