from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import typer

from irl.visualization.data import aggregate_runs
from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import (
    DPI,
    LEGEND_FRAMEALPHA,
    LEGEND_FONTSIZE,
    alpha_for_method,
    apply_grid,
    draw_order,
    linestyle_for_method,
    linewidth_for_method,
    zorder_for_method,
)


def _meta_tags(*, smooth: int, align: str) -> tuple[str, str, str]:
    a = str(align).strip().lower() or "interpolate"
    title_tag = f"smooth={int(smooth)}, align={a}"
    file_tag = f"smooth{int(smooth)}__align{a}"
    return a, title_tag, file_tag


def _env_tag(env_id: str) -> str:
    return str(env_id).replace("/", "-")


def _generate_comparison_plot(
    groups_by_env: Dict[str, Dict[str, List[Path]]],
    methods_to_plot: List[str],
    metric: str,
    smooth: int,
    shade: bool,
    title: str,
    filename_suffix: str,
    plots_root: Path,
    *,
    paper_mode: bool = False,
    align: str = "interpolate",
) -> None:
    _ = paper_mode
    align_mode, meta_title, meta_file = _meta_tags(smooth=int(smooth), align=str(align))

    plt = apply_rcparams_paper()

    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: kv[0]):
        relevant_methods = [m for m in methods_to_plot if m in by_method]
        if not relevant_methods:
            continue

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=int(DPI))
        any_plotted = False

        for method in draw_order(relevant_methods):
            dirs = by_method[method]
            try:
                agg = aggregate_runs(dirs, metric=metric, smooth=int(smooth), align=align_mode)
            except Exception:
                continue

            if agg.n_runs == 0 or agg.steps.size == 0:
                continue

            ax.plot(
                agg.steps,
                agg.mean,
                label=f"{method} (n={agg.n_runs})",
                linewidth=float(linewidth_for_method(method)),
                linestyle=linestyle_for_method(method),
                alpha=float(alpha_for_method(method)),
                zorder=int(zorder_for_method(method)),
                color=_color_for_method(method),
            )

            if shade and agg.n_runs > 1 and agg.steps.size > 0:
                ci = 1.96 * (agg.std / (float(agg.n_runs) ** 0.5))
                ax.fill_between(
                    agg.steps,
                    agg.mean - ci,
                    agg.mean + ci,
                    alpha=0.12,
                    color=_color_for_method(method),
                    linewidth=0.0,
                    zorder=max(0, int(zorder_for_method(method)) - 1),
                )

            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Environment steps")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{env_id} — {title} ({meta_title})")
        apply_grid(ax)
        ax.legend(loc="lower right", framealpha=float(LEGEND_FRAMEALPHA), fontsize=int(LEGEND_FONTSIZE))

        out = Path(plots_root) / f"{_env_tag(env_id)}__{filename_suffix}__{meta_file}.png"
        fig.tight_layout()
        save_fig_atomic(fig, out)
        plt.close(fig)
        typer.echo(f"[suite] Saved {filename_suffix} plot: {out}")
