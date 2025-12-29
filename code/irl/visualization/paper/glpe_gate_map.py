from __future__ import annotations

from pathlib import Path

import numpy as np

from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import DPI, FIGSIZE, LEGEND_FRAMEALPHA, LEGEND_FONTSIZE, apply_grid
from irl.visualization.trajectory_projection import trajectory_projection

from .glpe_common import sample_idx, sample_seed, select_latest_glpe_trajectories


def _env_tag(env_id: str) -> str:
    return str(env_id).replace("/", "-")


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

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)
    plt = apply_rcparams_paper()

    written: list[Path] = []

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

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=int(DPI))

        glpe_c = _color_for_method("glpe")

        if bool((~g).any()):
            ax.scatter(
                x[~g],
                y[~g],
                c="lightgray",
                s=18,
                alpha=0.55,
                edgecolor="none",
                label="Gated/Off",
                zorder=2,
            )
        if bool(g.any()):
            ax.scatter(
                x[g],
                y[g],
                c=glpe_c,
                s=22,
                alpha=0.85,
                edgecolor="none",
                label="Active/On",
                zorder=10,
            )

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"{env_id} — GLPE gate map (eval)")

        apply_grid(ax)
        ax.legend(loc="lower right", framealpha=float(LEGEND_FRAMEALPHA), fontsize=int(LEGEND_FONTSIZE))

        fig.tight_layout()
        out = plots_root / f"{_env_tag(env_id)}__glpe_gate_map.png"
        save_fig_atomic(fig, out)
        plt.close(fig)
        written.append(out)

    return written
