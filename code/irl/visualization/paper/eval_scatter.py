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
    apply_grid,
    legend_order,
)


def _env_tag(env_id: str) -> str:
    return str(env_id).replace("/", "-")


def _load_summary_raw(path: Path) -> pd.DataFrame | None:
    p = Path(path)
    try:
        df = pd.read_csv(p)
    except Exception:
        return None

    required = {"method", "env_id", "seed", "ckpt_step", "mean_return"}
    if not required.issubset(set(df.columns)):
        return None

    out = df.copy()
    out["env_id"] = out["env_id"].astype(str).str.strip()
    out["method"] = out["method"].astype(str).str.strip()
    out["method_key"] = out["method"].str.lower().str.strip()

    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    out["ckpt_step"] = pd.to_numeric(out["ckpt_step"], errors="coerce")
    out["mean_return"] = pd.to_numeric(out["mean_return"], errors="coerce")
    out = out.dropna(subset=["env_id", "method_key", "seed", "ckpt_step", "mean_return"]).copy()

    out["seed"] = out["seed"].astype(int)
    out["ckpt_step"] = out["ckpt_step"].astype(int)

    if "policy_mode" in out.columns:
        out["policy_mode"] = out["policy_mode"].astype(str).str.strip().str.lower()
        if "mode" in set(out["policy_mode"].unique().tolist()):
            out = out.loc[out["policy_mode"] == "mode"].copy()

    return out


def plot_eval_scatter_by_env(
    summary_raw_csv: Path,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str] | None,
    title: str,
    filename_suffix: str,
    alpha: float = 0.65,
    point_size: float = 18.0,
) -> list[Path]:
    raw_df = _load_summary_raw(Path(summary_raw_csv))
    if raw_df is None or raw_df.empty:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    want: list[str] | None
    if methods_to_plot is None:
        want = None
    else:
        norm = [str(m).strip().lower() for m in methods_to_plot if str(m).strip()]
        want = norm or None

    label_by_key = (
        raw_df.drop_duplicates(subset=["method_key"], keep="first")
        .set_index("method_key")["method"]
        .astype(str)
        .to_dict()
    )

    plt = apply_rcparams_paper()
    written: list[Path] = []

    for env_id in sorted(raw_df["env_id"].unique().tolist()):
        df_env = raw_df.loc[raw_df["env_id"] == env_id].copy()
        if df_env.empty:
            continue

        methods_present = sorted(set(df_env["method_key"].tolist()))
        methods = methods_present if want is None else [m for m in want if m in set(methods_present)]
        if not methods:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=int(DPI))

        for mk in methods:
            df_m = df_env.loc[df_env["method_key"] == mk].copy()
            if df_m.empty:
                continue

            x = pd.to_numeric(df_m["ckpt_step"], errors="coerce").to_numpy(dtype=np.float64)
            y = pd.to_numeric(df_m["mean_return"], errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(x) & np.isfinite(y)
            if not bool(ok.any()):
                continue
            x = x[ok]
            y = y[ok]

            is_glpe = str(mk).strip().lower() == "glpe"
            a = 1.0 if is_glpe else float(alpha)
            s = float(point_size) * (1.25 if is_glpe else 1.0)
            z = 10 if is_glpe else 2

            ax.scatter(
                x,
                y,
                s=float(s),
                alpha=float(a),
                color=_color_for_method(mk),
                edgecolor="none",
                label=str(label_by_key.get(mk, mk)),
                zorder=z,
            )

        ax.set_xlabel("Checkpoint step (env steps)")
        ax.set_ylabel("Mean episode return")
        ax.set_title(f"{env_id} — {title}")

        apply_grid(ax)

        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            by_label = {str(l): h for h, l in zip(handles, labels)}
            desired = []
            for mk in legend_order(methods):
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

        out = plots_root / f"{_env_tag(env_id)}__eval_scatter__{filename_suffix}.png"
        save_fig_atomic(fig, out)
        plt.close(fig)
        written.append(out)

    return written
