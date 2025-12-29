from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import DPI, FIGSIZE, LEGEND_FRAMEALPHA, LEGEND_FONTSIZE, apply_grid


def plot_normalized_summary(
    summary_path: Path,
    out_path: Path,
    highlight_method: str | None = "glpe",
    baseline_method: str = "vanilla",
    *,
    baseline_required: bool = False,
) -> None:
    if not Path(summary_path).exists():
        return

    try:
        df = pd.read_csv(Path(summary_path))
    except Exception:
        return

    required = {"method", "env_id", "mean_return_mean", "n_seeds"}
    if not required.issubset(set(df.columns)):
        return

    df = df.copy()
    df["env_id"] = df["env_id"].astype(str).str.strip()
    df["method"] = df["method"].astype(str).str.strip()
    df["method_key"] = df["method"].str.lower().str.strip()

    highlight_key = None if highlight_method is None else str(highlight_method).strip().lower() or None
    baseline_key = str(baseline_method).strip().lower()

    methods_present = set(df["method_key"].unique().tolist())
    if baseline_key not in methods_present:
        if baseline_required:
            return
        counts = df.groupby("method_key")["env_id"].nunique().sort_values(ascending=False)
        if counts.empty:
            return
        baseline_key = str(counts.index[0])

    label_by_key = (
        df.drop_duplicates(subset=["method_key"], keep="first")
        .set_index("method_key")["method"]
        .astype(str)
        .to_dict()
    )

    base_rows = df.loc[df["method_key"] == baseline_key, ["env_id", "mean_return_mean"]].copy()
    base_rows = base_rows.drop_duplicates(subset=["env_id"], keep="last").set_index("env_id")
    baseline_by_env = base_rows["mean_return_mean"].to_dict()
    envs = [str(e) for e in baseline_by_env.keys() if str(e).strip()]
    if not envs:
        return
    envs = sorted(envs)

    df2 = df[df["env_id"].isin(envs)].copy()
    df2["baseline_mean"] = df2["env_id"].map(baseline_by_env)
    df2["baseline_denom"] = np.maximum(1.0, np.abs(df2["baseline_mean"].astype(float)))
    df2["score_mean"] = (df2["mean_return_mean"].astype(float) - df2["baseline_mean"]) / df2[
        "baseline_denom"
    ]

    has_ci = {"mean_return_ci95_lo", "mean_return_ci95_hi"}.issubset(set(df2.columns))
    if has_ci:
        df2["score_ci_lo"] = (
            df2["mean_return_ci95_lo"].astype(float) - df2["baseline_mean"]
        ) / df2["baseline_denom"]
        df2["score_ci_hi"] = (
            df2["mean_return_ci95_hi"].astype(float) - df2["baseline_mean"]
        ) / df2["baseline_denom"]
    elif "mean_return_se" in df2.columns:
        z = 1.96
        lo = df2["mean_return_mean"].astype(float) - z * df2["mean_return_se"].astype(float)
        hi = df2["mean_return_mean"].astype(float) + z * df2["mean_return_se"].astype(float)
        df2["score_ci_lo"] = (lo - df2["baseline_mean"]) / df2["baseline_denom"]
        df2["score_ci_hi"] = (hi - df2["baseline_mean"]) / df2["baseline_denom"]
        has_ci = True
    else:
        df2["score_ci_lo"] = np.nan
        df2["score_ci_hi"] = np.nan

    score = df2.pivot_table(index="env_id", columns="method_key", values="score_mean", aggfunc="mean")
    score = score.reindex(envs)
    if score.empty:
        return

    ci_lo = None
    ci_hi = None
    if has_ci:
        ci_lo = df2.pivot_table(index="env_id", columns="method_key", values="score_ci_lo", aggfunc="mean")
        ci_hi = df2.pivot_table(index="env_id", columns="method_key", values="score_ci_hi", aggfunc="mean")
        ci_lo = ci_lo.reindex(score.index)
        ci_hi = ci_hi.reindex(score.index)

    n_seeds = df2.pivot_table(index="env_id", columns="method_key", values="n_seeds", aggfunc="max")
    n_seeds = n_seeds.reindex(score.index)

    methods_all = sorted([str(m) for m in score.columns.tolist() if str(m).strip()])
    ordered: list[str] = []
    if baseline_key in methods_all:
        ordered.append(baseline_key)
    for m in methods_all:
        if m == baseline_key:
            continue
        if highlight_key is not None and m == highlight_key:
            continue
        ordered.append(m)
    if highlight_key is not None and highlight_key in methods_all and highlight_key not in ordered:
        ordered.append(highlight_key)

    score = score.reindex(columns=ordered)
    if ci_lo is not None and ci_hi is not None:
        ci_lo = ci_lo.reindex(columns=ordered)
        ci_hi = ci_hi.reindex(columns=ordered)
    n_seeds = n_seeds.reindex(columns=ordered)

    n_envs = int(score.shape[0])
    n_methods = int(score.shape[1])
    if n_envs <= 0 or n_methods <= 0:
        return

    plt = apply_rcparams_paper()

    width = 0.8 / float(n_methods)
    x = np.arange(n_envs, dtype=np.float64)

    fig_w = max(float(FIGSIZE[0]), 1.1 * float(n_envs))
    fig, ax = plt.subplots(figsize=(fig_w, float(FIGSIZE[1])), dpi=int(DPI))

    ax.axhline(0.0, linewidth=1.0, alpha=0.6, color="black")

    y_min = float(np.nanmin(ci_lo.to_numpy())) if ci_lo is not None else float(np.nanmin(score.to_numpy()))
    y_max = float(np.nanmax(ci_hi.to_numpy())) if ci_hi is not None else float(np.nanmax(score.to_numpy()))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = -1.0, 1.0
    y_min = min(y_min, 0.0)
    y_max = max(y_max, 0.0)
    span = max(1e-6, float(y_max - y_min))
    pad = 0.12 * span
    txt_off = 0.02 * span

    for i, mk in enumerate(ordered):
        vals = score[mk].to_numpy(dtype=np.float64, copy=False)
        offset = (float(i) - float(n_methods) / 2.0) * width + width / 2.0
        xpos = x + offset

        finite = np.isfinite(vals)
        if not bool(finite.any()):
            continue

        y = vals[finite]
        x_ok = xpos[finite]

        yerr = None
        if ci_lo is not None and ci_hi is not None and mk in ci_lo.columns:
            lo = ci_lo[mk].to_numpy(dtype=np.float64, copy=False)[finite]
            hi = ci_hi[mk].to_numpy(dtype=np.float64, copy=False)[finite]
            lower = np.maximum(0.0, y - lo)
            upper = np.maximum(0.0, hi - y)
            yerr = np.vstack([lower, upper])

        alpha = 1.0 if mk == (highlight_key or "") else 0.88
        zorder = 10 if mk == (highlight_key or "") else 2

        ax.bar(
            x_ok,
            y,
            width,
            color=_color_for_method(mk),
            alpha=float(alpha),
            edgecolor="none",
            linewidth=0.0,
            yerr=yerr,
            capsize=2 if yerr is not None else 0,
            zorder=zorder,
            label=str(label_by_key.get(mk, mk)),
        )

        if mk in n_seeds.columns:
            nvals = n_seeds[mk].to_numpy(dtype=np.float64, copy=False)[finite]
            for xx, yy, nn in zip(x_ok.tolist(), y.tolist(), nvals.tolist()):
                if not np.isfinite(nn):
                    continue
                n_int = int(round(float(nn)))
                ax.text(
                    float(xx),
                    float(yy + txt_off) if yy >= 0.0 else float(yy - txt_off),
                    f"n={n_int}",
                    ha="center",
                    va="bottom" if yy >= 0.0 else "top",
                    fontsize=LEGEND_FONTSIZE,
                    alpha=0.9,
                    zorder=20,
                )

    ax.set_ylabel(f"Return vs baseline {label_by_key.get(baseline_key, baseline_key)} (Δ / max(1, |baseline|))")
    ax.set_title("Baseline-relative performance (mean return)")
    ax.set_xticks(x)
    ax.set_xticklabels(score.index.tolist(), rotation=30, ha="right")
    ax.set_ylim(float(y_min - pad), float(y_max + pad))

    apply_grid(ax)

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(
            handles,
            labels,
            loc="lower right",
            framealpha=float(LEGEND_FRAMEALPHA),
            fontsize=int(LEGEND_FONTSIZE),
        )

    fig.tight_layout()
    save_fig_atomic(fig, Path(out_path))
    plt.close(fig)
