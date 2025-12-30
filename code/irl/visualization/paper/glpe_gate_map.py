from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import DPI, FIGSIZE, LEGEND_FRAMEALPHA, LEGEND_FONTSIZE, apply_grid
from irl.visualization.trajectory_projection import trajectory_projection

from .glpe_common import sample_idx, sample_seed, select_latest_glpe_trajectories


def _grid(n: int) -> tuple[int, int]:
    nn = int(n)
    if nn <= 0:
        return 0, 0
    if nn == 1:
        return 1, 1
    ncols = 2
    nrows = int(math.ceil(float(nn) / float(ncols)))
    return nrows, ncols


def _figsize(nrows: int, ncols: int) -> tuple[float, float]:
    base_w, base_h = float(FIGSIZE[0]), float(FIGSIZE[1])
    w = base_w if int(ncols) <= 1 else base_w * 1.75
    h = base_h * float(max(1, int(nrows)))
    return float(w), float(h)


def plot_glpe_state_gate_map(
    *,
    traj_root: Path,
    plots_root: Path,
    max_points: int = 40000,
) -> list[Path]:
    traj_root = Path(traj_root)
    if not traj_root.exists():
        return []

    selected = select_latest_glpe_trajectories(traj_root)
    by_env: dict[str, list[Path]] = {}
    for env_id, _run_name, _ckpt_step, p in selected:
        by_env.setdefault(str(env_id), []).append(Path(p))

    if not by_env:
        return []

    env_recs: list[tuple[str, np.ndarray, np.ndarray, str, str]] = []

    for env_id, paths in sorted(by_env.items(), key=lambda kv: str(kv[0])):
        obs_all: list[np.ndarray] = []
        gates_all: list[np.ndarray] = []

        for p in paths:
            try:
                data = np.load(p, allow_pickle=False)
                obs = np.asarray(data["obs"], dtype=np.float32)
                gates = np.asarray(data["gates"]).reshape(-1)
            except Exception:
                continue
            if obs.ndim != 2 or gates.size != obs.shape[0]:
                continue
            obs_all.append(obs)
            gates_all.append(gates.astype(np.float32, copy=False))

        if not obs_all:
            continue

        obs_cat = np.concatenate(obs_all, axis=0)
        gates_cat = np.concatenate(gates_all, axis=0)

        proj = trajectory_projection(env_id, obs_cat, include_bipedalwalker=True)
        if proj is None:
            continue

        x, y, xlab, ylab, _proj_note = proj
        g = gates_cat >= 0.5

        finite = np.isfinite(x) & np.isfinite(y)
        if not bool(finite.any()):
            continue

        x = np.asarray(x[finite], dtype=np.float64)
        y = np.asarray(y[finite], dtype=np.float64)
        g = np.asarray(g[finite], dtype=bool)

        idx = sample_idx(x.shape[0], int(max_points), seed=sample_seed("glpe_gate_map", env_id))
        x = x[idx]
        y = y[idx]
        g = g[idx]

        env_recs.append((str(env_id), x, y, str(xlab), str(ylab)))

    if not env_recs:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)
    plt = apply_rcparams_paper()

    nrows, ncols = _grid(len(env_recs))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=_figsize(nrows, ncols),
        dpi=int(DPI),
        squeeze=False,
    )
    axes_flat = list(axes.reshape(-1))

    glpe_c = _color_for_method("glpe")

    for i, (env_id, x, y, xlab, ylab) in enumerate(env_recs):
        ax = axes_flat[i]

        g = np.asarray(by_env[env_id], dtype=object)  # placeholder to satisfy type checkers
        _ = g  # avoid relying on stored bools; re-load from x/y masks below

        try:
            data_all: list[np.ndarray] = []
            gates_all: list[np.ndarray] = []
            for p in by_env.get(env_id, []):
                try:
                    d = np.load(p, allow_pickle=False)
                    obs = np.asarray(d["obs"], dtype=np.float32)
                    gates = np.asarray(d["gates"]).reshape(-1).astype(np.float32, copy=False)
                    if obs.ndim == 2 and gates.size == obs.shape[0]:
                        data_all.append(obs)
                        gates_all.append(gates)
                except Exception:
                    continue
            if data_all and gates_all:
                obs_cat = np.concatenate(data_all, axis=0)
                gates_cat = np.concatenate(gates_all, axis=0) >= 0.5
                proj = trajectory_projection(env_id, obs_cat, include_bipedalwalker=True)
                if proj is not None:
                    px, py, _, _, _ = proj
                    finite = np.isfinite(px) & np.isfinite(py)
                    px = np.asarray(px[finite], dtype=np.float64)
                    py = np.asarray(py[finite], dtype=np.float64)
                    gg = np.asarray(gates_cat[finite], dtype=bool)
                    idx = sample_idx(px.shape[0], int(max_points), seed=sample_seed("glpe_gate_map", env_id))
                    x = px[idx]
                    y = py[idx]
                    gate_on = gg[idx]
                else:
                    gate_on = np.ones((x.shape[0],), dtype=bool)
            else:
                gate_on = np.ones((x.shape[0],), dtype=bool)
        except Exception:
            gate_on = np.ones((x.shape[0],), dtype=bool)

        gated = ~gate_on
        if bool(gated.any()):
            ax.scatter(
                x[gated],
                y[gated],
                c="lightgray",
                s=18,
                alpha=0.55,
                edgecolor="none",
                linewidth=0.0,
                zorder=2,
            )

        if bool(gate_on.any()):
            ax.scatter(
                x[gate_on],
                y[gate_on],
                c=glpe_c,
                s=22,
                alpha=0.85,
                edgecolor="none",
                linewidth=0.0,
                zorder=10,
            )

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(str(env_id))
        apply_grid(ax)

    for j in range(len(env_recs), len(axes_flat)):
        try:
            axes_flat[j].axis("off")
        except Exception:
            pass

    handles = [
        plt.Line2D([], [], color="lightgray", marker="o", linestyle="none", markersize=6),
        plt.Line2D([], [], color=glpe_c, marker="o", linestyle="none", markersize=6),
    ]
    labels = ["Gated/Off", "Active/On"]

    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        framealpha=float(LEGEND_FRAMEALPHA),
        fontsize=int(LEGEND_FONTSIZE),
    )

    fig.suptitle("GLPE gate map (eval)")
    fig.tight_layout(rect=[0.0, 0.07, 1.0, 0.94])

    out = plots_root / "glpe_gate_map__eval.png"
    save_fig_atomic(fig, out)
    plt.close(fig)
    return [out]
