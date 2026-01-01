from __future__ import annotations

import re
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


def _patch_matplotlib_tight_layout_defaults() -> None:
    import matplotlib.figure

    if bool(getattr(matplotlib.figure.Figure, "_irl_tight_layout_patched", False)):
        return

    orig_tight_layout = matplotlib.figure.Figure.tight_layout

    def _tight_layout(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if args:
            return orig_tight_layout(self, *args, **kwargs)

        from irl.visualization.labels import (
            LEGEND_TIGHT_LAYOUT_PAD_MULT,
            TIGHT_LAYOUT_H_PAD_MULT,
            TIGHT_LAYOUT_W_PAD_MULT,
        )

        rect = kwargs.pop("rect", None)

        h_pad_in = "h_pad" in kwargs
        w_pad_in = "w_pad" in kwargs

        pad = kwargs.pop("pad", None)
        h_pad = kwargs.pop("h_pad", None)
        w_pad = kwargs.pop("w_pad", None)

        if pad is None:
            pad = float(LEGEND_TIGHT_LAYOUT_PAD_MULT)
        try:
            pad_f = float(pad)
        except Exception:
            pad_f = float(LEGEND_TIGHT_LAYOUT_PAD_MULT)

        if not h_pad_in and h_pad is None and TIGHT_LAYOUT_H_PAD_MULT is not None:
            try:
                h_pad = float(TIGHT_LAYOUT_H_PAD_MULT)
            except Exception:
                h_pad = None

        if not w_pad_in and w_pad is None and TIGHT_LAYOUT_W_PAD_MULT is not None:
            try:
                w_pad = float(TIGHT_LAYOUT_W_PAD_MULT)
            except Exception:
                w_pad = None

        return orig_tight_layout(self, pad=pad_f, h_pad=h_pad, w_pad=w_pad, rect=rect)

    matplotlib.figure.Figure.tight_layout = _tight_layout  # type: ignore[assignment]
    setattr(matplotlib.figure.Figure, "_irl_tight_layout_patched", True)


def apply_rcparams_paper():
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    plt.rcParams.update(_STYLE_RCPARAMS)
    _disable_matplotlib_titles()
    _patch_matplotlib_tight_layout_defaults()
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


_ENV_ORDER_BASE: tuple[str, ...] = (
    "mountaincar",
    "bipedalwalker",
    "halfcheetah",
    "ant",
    "carracing",
    "humanoid",
)
_ENV_RANK: dict[str, int] = {env: i for i, env in enumerate(_ENV_ORDER_BASE)}
_ENV_VERSION_RE = re.compile(r"-v\d+$", re.IGNORECASE)


def _env_base(env_id: str) -> str:
    s = str(env_id).strip().replace("/", "-")
    s = _ENV_VERSION_RE.sub("", s)
    return s.lower()


def env_sort_key(env_id: object) -> tuple[int, str, str]:
    s = str(env_id).strip()
    if not s:
        return (10_000, "", "")
    key = s.replace("/", "-")
    base = _env_base(key)
    rank = int(_ENV_RANK.get(base, 10_000))
    return (rank, base, key.lower())


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
