from __future__ import annotations

from pathlib import Path

import numpy as np

from irl.visualization.labels import add_legend_rows_top, add_row_label, env_label, legend_ncol, method_label
from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import DPI, FIG_WIDTH, LEGEND_FONTSIZE, apply_grid
from irl.visualization.trajectory_projection import trajectory_projection

from .glpe_common import sample_idx, sample_seed, select_latest_glpe_trajectories


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

    env_recs: list[dict[str, object]] = []
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
        gates_cat = (np.concatenate(gates_all, axis=0) >= 0.5).astype(bool, copy=False)

        proj = trajectory_projection(env_id, obs_cat, include_bipedalwalker=True)
        if proj is None:
            continue

        x, y, xlab, ylab, _proj_note = proj
        finite = np.isfinite(x) & np.isfinite(y)
        if not bool(finite.any()):
            continue

        x = np.asarray(x[finite], dtype=np.float64)
        y = np.asarray(y[finite], dtype=np.float64)
        gate_on = np.asarray(gates_cat[finite], dtype=bool)

        idx = sample_idx(x.shape[0], int(max_points), seed=sample_seed("glpe_gate_map", env_id))
        x = x[idx]
        y = y[idx]
        gate_on = gate_on[idx]

        env_recs.append(
            {
                "env_id": str(env_id),
                "x": x,
                "y": y,
                "gate_on": gate_on,
                "xlab": str(xlab),
                "ylab": str(ylab),
            }
        )

    if not env_recs:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)
    plt = apply_rcparams_paper()

    nrows = int(len(env_recs))
    height = max(2.8, 2.2 * float(nrows))

    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(float(FIG_WIDTH), float(height)),
        dpi=int(DPI),
        squeeze=False,
    )

    glpe_c = _color_for_method("glpe")

    for i, rec in enumerate(env_recs):
        ax = axes[i, 0]
        env_id = str(rec["env_id"])
        x = np.asarray(rec["x"], dtype=np.float64)
        y = np.asarray(rec["y"], dtype=np.float64)
        gate_on = np.asarray(rec["gate_on"], dtype=bool)

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

        ax.set_xlabel(str(rec["xlab"]) if i == nrows - 1 else "")
        ax.set_ylabel(str(rec["ylab"]))
        if i != nrows - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)

        apply_grid(ax)
        add_row_label(ax, env_label(env_id))

    handles = [
        plt.Line2D([], [], color="lightgray", marker="o", linestyle="none", markersize=6),
        plt.Line2D([], [], color=glpe_c, marker="o", linestyle="none", markersize=6),
    ]
    labels = ["Gate off", "Gate on"]

    top = add_legend_rows_top(fig, [(handles, labels, legend_ncol(len(handles)))], fontsize=int(LEGEND_FONTSIZE))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, float(top)])

    out = plots_root / "glpe-gate-map.png"
    save_fig_atomic(fig, out)
    plt.close(fig)
    return [out]
