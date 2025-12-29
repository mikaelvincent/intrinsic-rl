from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import (
    DPI,
    FIGSIZE,
    LEGEND_FRAMEALPHA,
    LEGEND_FONTSIZE,
    alpha_for_method,
    apply_grid,
    draw_order,
    legend_order,
    linestyle_for_method,
    linewidth_for_method,
    zorder_for_method,
)
from .thresholds import add_solved_threshold_line


def _env_tag(env_id: str) -> str:
    return str(env_id).replace("/", "-")


def plot_eval_curves_by_env(
    by_step_df: pd.DataFrame,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
) -> list[Path]:
    if by_step_df is None or by_step_df.empty:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    want = [str(m).strip().lower() for m in methods_to_plot if str(m).strip()]
    if not want:
        return []

    df = by_step_df.copy()
    df["env_id"] = df["env_id"].astype(str).str.strip()
    if "method_key" not in df.columns:
        df["method_key"] = df["method"].astype(str).str.strip().str.lower()
    df["method_key"] = df["method_key"].astype(str).str.strip().str.lower()

    label_by_key = (
        df.drop_duplicates(subset=["method_key"], keep="first")
        .set_index("method_key")["method"]
        .astype(str)
        .to_dict()
        if "method" in df.columns
        else {}
    )

    plt = apply_rcparams_paper()
    written: list[Path] = []

    for env_id in sorted(df["env_id"].unique().tolist()):
        df_env = df.loc[df["env_id"] == env_id].copy()
        if df_env.empty:
            continue

        methods_present = sorted(set(df_env["method_key"].tolist()) & set(want))
        if not methods_present:
            continue

        uniq_steps = sorted(set(df_env["ckpt_step"].tolist()))
        if len(uniq_steps) <= 1:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=int(DPI))

        methods_draw = draw_order([m for m in want if m in set(methods_present)])
        for mk in methods_draw:
            df_m = df_env.loc[df_env["method_key"] == mk].copy()
            if df_m.empty:
                continue
            df_m = df_m.sort_values("ckpt_step").drop_duplicates(subset=["ckpt_step"], keep="last")

            x = df_m["ckpt_step"].to_numpy(dtype=np.int64, copy=False)
            y = pd.to_numeric(df_m["mean_return_mean"], errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(x.astype(np.float64)) & np.isfinite(y)
            if not bool(ok.any()):
                continue
            x = x[ok]
            y = y[ok]

            ax.plot(
                x,
                y,
                color=_color_for_method(mk),
                lw=float(linewidth_for_method(mk)),
                ls=linestyle_for_method(mk),
                alpha=float(alpha_for_method(mk)),
                zorder=int(zorder_for_method(mk)),
                label=str(label_by_key.get(mk, mk)),
            )

        add_solved_threshold_line(ax, str(env_id))

        ax.set_xlabel("Checkpoint step (env steps)")
        ax.set_ylabel("Mean episode return")
        ax.set_title(f"{env_id} — {title}")

        apply_grid(ax)

        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            by_label = {str(l): h for h, l in zip(handles, labels)}
            desired = []
            for mk in legend_order([m for m in want if m in set(methods_present)]):
                lbl = str(label_by_key.get(mk, mk))
                if lbl in by_label:
                    desired.append((by_label[lbl], lbl))
            if desired:
                ax.legend(
                    [h for h, _ in desired],
                    [l for _, l in desired],
                    loc="lower right",
                    framealpha=float(LEGEND_FRAMEALPHA),
                    fontsize=int(LEGEND_FONTSIZE),
                )

        fig.tight_layout()

        out = plots_root / f"{_env_tag(env_id)}__{filename_suffix}.png"
        save_fig_atomic(fig, out)
        plt.close(fig)
        written.append(out)

    return written
