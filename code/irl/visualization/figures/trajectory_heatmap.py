from __future__ import annotations

from pathlib import Path

import numpy as np

from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import DPI, FIGSIZE, LEGEND_FRAMEALPHA, LEGEND_FONTSIZE, apply_grid
from irl.visualization.trajectory_projection import trajectory_projection


def _as_str_scalar(x: object) -> str | None:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        return str(arr.reshape(-1)[0])
    except Exception:
        return None


def plot_trajectory_heatmap(npz_path: Path, out_path: Path, max_points: int = 20000) -> bool:
    if not Path(npz_path).exists():
        return False

    try:
        data = np.load(Path(npz_path), allow_pickle=True)
        obs = data["obs"]
        gates = data["gates"]
    except Exception:
        return False

    env_id = _as_str_scalar(data.get("env_id")) if hasattr(data, "get") else None
    method = _as_str_scalar(data.get("method")) if hasattr(data, "get") else None
    gate_source = _as_str_scalar(data.get("gate_source")) if hasattr(data, "get") else None

    env_disp = env_id or Path(npz_path).stem.replace("_trajectory", "")
    obs_arr = np.asarray(obs)
    if obs_arr.ndim != 2:
        return False

    proj = trajectory_projection(env_id, obs_arr)
    if proj is None:
        return False

    x, y, xlab, ylab, proj_note = proj

    g = np.asarray(gates).reshape(-1)
    if int(g.size) != int(x.shape[0]):
        return False

    n = int(x.shape[0])
    if n > int(max_points):
        idx = np.linspace(0, n - 1, int(max_points), dtype=int)
        x = np.asarray(x[idx], dtype=np.float64)
        y = np.asarray(y[idx], dtype=np.float64)
        g = np.asarray(g[idx], dtype=np.float32)
    else:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        g = np.asarray(g, dtype=np.float32)

    finite = np.isfinite(x) & np.isfinite(y)
    if not bool(finite.any()):
        return False

    x = x[finite]
    y = y[finite]
    g = g[finite]

    active = g >= 0.5
    gated = ~active

    gate_note = (gate_source or "recomputed").strip().lower()
    if gate_note not in {"checkpoint", "recomputed", "mixed", "n/a", "missing_intrinsic"}:
        gate_note = "recomputed"

    title_bits: list[str] = [str(env_disp)]
    if method:
        title_bits.append(str(method))
    title_bits.append(f"gates: {gate_note}")

    method_key = (str(method).strip().lower() if method is not None else "").strip() or "glpe"
    active_color = _color_for_method(method_key)

    plt = apply_rcparams_paper()
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=int(DPI))

    if bool(gated.any()):
        ax.scatter(
            x[gated],
            y[gated],
            c="lightgray",
            s=9,
            alpha=0.5,
            edgecolor="none",
            label="Gated/Off",
            zorder=2,
        )

    if bool(active.any()):
        ax.scatter(
            x[active],
            y[active],
            c=active_color,
            s=11,
            alpha=0.5,
            edgecolor="none",
            label="Active/On",
            zorder=10,
        )

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(" â€” ".join(title_bits))

    apply_grid(ax)

    ax.legend(loc="lower right", framealpha=float(LEGEND_FRAMEALPHA), fontsize=int(LEGEND_FONTSIZE))

    if proj_note:
        fig.text(0.01, 0.01, str(proj_note), ha="left", va="bottom", fontsize=LEGEND_FONTSIZE, alpha=0.9)

    fig.tight_layout()
    save_fig_atomic(fig, Path(out_path))
    plt.close(fig)
    return True
