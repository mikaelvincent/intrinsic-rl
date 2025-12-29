from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import typer

from irl.visualization.data import aggregate_runs
from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import DPI, LEGEND_FRAMEALPHA, LEGEND_FONTSIZE, apply_grid


def _meta_tags(*, smooth: int, align: str) -> tuple[str, str, str]:
    a = str(align).strip().lower() or "interpolate"
    title_tag = f"smooth={int(smooth)}, align={a}"
    file_tag = f"smooth{int(smooth)}__align{a}"
    return a, title_tag, file_tag


def _env_tag(env_id: str) -> str:
    return str(env_id).replace("/", "-")


def _generate_gating_plot(
    groups_by_env: Dict[str, Dict[str, List[Path]]],
    plots_root: Path,
    smooth: int = 25,
    *,
    align: str = "interpolate",
) -> None:
    align_mode, meta_title, meta_file = _meta_tags(smooth=int(smooth), align=str(align))
    plt = apply_rcparams_paper()

    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: kv[0]):
        runs = by_method.get("glpe")
        if not runs:
            for m, r in by_method.items():
                if m.lower() == "glpe":
                    runs = r
                    break
        if not runs:
            continue

        try:
            agg_rew = aggregate_runs(runs, metric="reward_mean", smooth=smooth, align=align_mode)
            agg_gate = aggregate_runs(runs, metric="gate_rate", smooth=smooth, align=align_mode)
            if agg_gate.n_runs == 0:
                agg_gate = aggregate_runs(runs, metric="gate_rate_pct", smooth=smooth, align=align_mode)
        except Exception:
            continue

        if agg_rew.n_runs == 0 or agg_gate.n_runs == 0:
            continue

        fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=int(DPI))

        c_rew = _color_for_method("vanilla")
        c_gate = _color_for_method("glpe")

        ax1.plot(agg_rew.steps, agg_rew.mean, color=c_rew, linewidth=1.9, alpha=0.88, label="Reward")
        ax1.set_xlabel("Environment steps")
        ax1.set_ylabel("Extrinsic reward")

        ax2 = ax1.twinx()
        is_pct = float(getattr(agg_gate.mean, "max", lambda: 0.0)()) > 1.1 if hasattr(agg_gate, "mean") else False
        ylabel = "Gate rate (%)" if is_pct else "Gate rate (0–1)"
        if not is_pct:
            ax2.set_ylim(0, 1.05)

        ax2.plot(
            agg_gate.steps,
            agg_gate.mean,
            color=c_gate,
            linewidth=3.6,
            alpha=1.0,
            linestyle="-",
            label="Gate rate",
        )
        ax2.set_ylabel(ylabel)

        ax1.set_title(f"{env_id} — Gating dynamics (GLPE) ({meta_title})")
        apply_grid(ax1)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax1.legend(
                list(h2) + list(h1),
                list(l2) + list(l1),
                loc="lower right",
                framealpha=float(LEGEND_FRAMEALPHA),
                fontsize=int(LEGEND_FONTSIZE),
            )

        out = Path(plots_root) / f"{_env_tag(env_id)}__gating_dynamics__{meta_file}.png"
        fig.tight_layout()
        save_fig_atomic(fig, out)
        plt.close(fig)
        typer.echo(f"[suite] Saved gating plot: {out}")
