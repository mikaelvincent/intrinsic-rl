from __future__ import annotations

from pathlib import Path

import numpy as np

from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import DPI, FIGSIZE, LEGEND_FRAMEALPHA, LEGEND_FONTSIZE, apply_grid

from .glpe_common import sample_idx, sample_seed, select_latest_glpe_trajectories


def _env_tag(env_id: str) -> str:
    return str(env_id).replace("/", "-")


def plot_glpe_extrinsic_vs_intrinsic(
    *,
    traj_root: Path,
    plots_root: Path,
    max_points: int = 60000,
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
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        gs: list[np.ndarray] = []

        for p in paths:
            try:
                data = np.load(p, allow_pickle=False)
                if "rewards_ext" not in getattr(data, "files", []):
                    continue
                r_ext = np.asarray(data["rewards_ext"], dtype=np.float32).reshape(-1)
                r_int = np.asarray(data["intrinsic"], dtype=np.float32).reshape(-1)
                gates = np.asarray(data["gates"]).reshape(-1)
            except Exception:
                continue

            n = int(min(r_ext.size, r_int.size, gates.size))
            if n <= 0:
                continue

            xs.append(r_ext[:n])
            ys.append(r_int[:n])
            gs.append(gates[:n].astype(np.float32, copy=False))

        if not xs:
            continue

        x = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
        y = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
        g = (np.concatenate(gs, axis=0) >= 0.5).astype(bool, copy=False)

        idx = sample_idx(x.shape[0], int(max_points), seed=sample_seed("glpe_extint", env_id))
        x = x[idx]
        y = y[idx]
        g = g[idx]

        finite = np.isfinite(x.astype(np.float64)) & np.isfinite(y.astype(np.float64))
        if not bool(finite.any()):
            continue
        x = x[finite]
        y = y[finite]
        g = g[finite]

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=int(DPI))

        glpe_c = _color_for_method("glpe")

        if bool((~g).any()):
            ax.scatter(
                x[~g],
                y[~g],
                s=18,
                alpha=0.55,
                c="lightgray",
                edgecolor="none",
                label="Gated/Off",
                zorder=2,
            )
        if bool(g.any()):
            ax.scatter(
                x[g],
                y[g],
                s=22,
                alpha=0.85,
                c=glpe_c,
                edgecolor="none",
                label="Active/On",
                zorder=10,
            )

        ax.axhline(0.0, linewidth=1.0, alpha=0.35, color="black")
        ax.axvline(0.0, linewidth=1.0, alpha=0.35, color="black")

        ax.set_xlabel("Extrinsic reward (per step)")
        ax.set_ylabel("Intrinsic reward (per step)")
        ax.set_title(f"{env_id} — GLPE extrinsic vs intrinsic (eval)")

        apply_grid(ax)
        ax.legend(loc="lower right", framealpha=float(LEGEND_FRAMEALPHA), fontsize=int(LEGEND_FONTSIZE))

        fig.tight_layout()
        out = plots_root / f"{_env_tag(env_id)}__glpe_extrinsic_vs_intrinsic.png"
        save_fig_atomic(fig, out)
        plt.close(fig)
        written.append(out)

    return written
