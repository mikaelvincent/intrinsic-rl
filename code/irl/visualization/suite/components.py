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


def _generate_component_plot(
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
            agg_imp = aggregate_runs(runs, metric="impact_rms", smooth=smooth, align=align_mode)
            agg_lp = aggregate_runs(runs, metric="lp_rms", smooth=smooth, align=align_mode)
        except Exception:
            continue

        if agg_imp.n_runs == 0 or agg_lp.n_runs == 0:
            continue

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=int(DPI))

        ax.plot(
            agg_imp.steps,
            agg_imp.mean,
            label="Impact (novelty)",
            color=_color_for_method("vanilla"),
            linewidth=1.9,
            alpha=0.88,
        )
        ax.plot(
            agg_lp.steps,
            agg_lp.mean,
            label="LP (competence)",
            color=_color_for_method("icm"),
            linewidth=1.9,
            alpha=0.88,
        )

        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Running RMS")
        ax.set_title(f"{env_id} — Intrinsic component evolution (GLPE) ({meta_title})")

        apply_grid(ax)
        ax.legend(loc="lower right", framealpha=float(LEGEND_FRAMEALPHA), fontsize=int(LEGEND_FONTSIZE))

        out = Path(plots_root) / f"{_env_tag(env_id)}__component_evolution__{meta_file}.png"
        fig.tight_layout()
        save_fig_atomic(fig, out)
        plt.close(fig)
        typer.echo(f"[suite] Saved component plot: {out}")
