from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from irl.utils.checkpoint import atomic_replace
from .data import ensure_parent
from .palette import color_for_method as _color_for_method
from .trajectory_projection import trajectory_projection


def plot_normalized_summary(
    summary_path: Path,
    out_path: Path,
    highlight_method: str | None = "glpe",
    baseline_method: str = "vanilla",
    *,
    baseline_required: bool = False,
) -> None:
    if not summary_path.exists():
        return

    df = pd.read_csv(summary_path)
    required = {"method", "env_id", "mean_return_mean", "n_seeds"}
    if not required.issubset(df.columns):
        return

    df = df.copy()
    df["env_id"] = df["env_id"].astype(str).str.strip()
    df["method"] = df["method"].astype(str).str.strip()
    df["method_key"] = df["method"].str.lower()

    hk = "" if highlight_method is None else str(highlight_method).strip().lower()
    highlight_key: str | None = hk or None

    baseline_req = str(baseline_method).strip().lower()

    methods_present = set(df["method_key"].unique().tolist())
    if baseline_req not in methods_present:
        if baseline_required:
            warnings.warn(
                f"Baseline method {baseline_method!r} not found; skipping normalized summary.",
                UserWarning,
            )
            return

        counts = df.groupby("method_key")["env_id"].nunique().sort_values(ascending=False)
        if counts.empty:
            return
        baseline_key = str(counts.index[0])
        baseline_label = str(df.loc[df["method_key"] == baseline_key, "method"].iloc[0])
        warnings.warn(
            f"Baseline method {baseline_method!r} not found; using {baseline_label!r}.",
            UserWarning,
        )
    else:
        baseline_key = baseline_req
        baseline_label = str(df.loc[df["method_key"] == baseline_key, "method"].iloc[0])

    label_by_key = (
        df.drop_duplicates(subset=["method_key"], keep="first")
        .set_index("method_key")["method"]
        .to_dict()
    )

    base_rows = df.loc[df["method_key"] == baseline_key, ["env_id", "mean_return_mean"]].copy()
    base_rows = base_rows.drop_duplicates(subset=["env_id"], keep="last").set_index("env_id")
    baseline_by_env = base_rows["mean_return_mean"].to_dict()

    envs_with_baseline = sorted(baseline_by_env.keys())
    if not envs_with_baseline:
        return

    df2 = df[df["env_id"].isin(envs_with_baseline)].copy()
    df2["baseline_mean"] = df2["env_id"].map(baseline_by_env)
    df2["baseline_denom"] = np.maximum(1.0, np.abs(df2["baseline_mean"].astype(float)))
    df2["score_mean"] = (df2["mean_return_mean"].astype(float) - df2["baseline_mean"]) / df2[
        "baseline_denom"
    ]

    has_ci = {"mean_return_ci95_lo", "mean_return_ci95_hi"}.issubset(df2.columns)
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

    score = df2.pivot_table(
        index="env_id", columns="method_key", values="score_mean", aggfunc="mean"
    ).sort_index()

    if score.empty:
        return

    ci_lo = None
    ci_hi = None
    if has_ci:
        ci_lo = df2.pivot_table(
            index="env_id", columns="method_key", values="score_ci_lo", aggfunc="mean"
        ).reindex(score.index)
        ci_hi = df2.pivot_table(
            index="env_id", columns="method_key", values="score_ci_hi", aggfunc="mean"
        ).reindex(score.index)

    n_seeds = df2.pivot_table(
        index="env_id", columns="method_key", values="n_seeds", aggfunc="max"
    ).reindex(score.index)

    methods_all = sorted(score.columns.tolist())
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

    score = score[ordered]
    if ci_lo is not None and ci_hi is not None:
        ci_lo = ci_lo.reindex(columns=ordered)
        ci_hi = ci_hi.reindex(columns=ordered)
    n_seeds = n_seeds.reindex(columns=ordered)

    n_envs = int(score.shape[0])
    n_methods = int(score.shape[1])
    if n_envs == 0 or n_methods == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, n_envs * 2), 6))

    x = np.arange(n_envs, dtype=np.float64)
    width = 0.8 / float(n_methods)

    colors: dict[str, str] = {m: _color_for_method(m) for m in ordered}

    if ci_lo is not None and ci_hi is not None:
        y_min = float(np.nanmin(ci_lo.to_numpy()))
        y_max = float(np.nanmax(ci_hi.to_numpy()))
    else:
        y_min = float(np.nanmin(score.to_numpy()))
        y_max = float(np.nanmax(score.to_numpy()))

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = -1.0, 1.0

    y_min = min(y_min, 0.0)
    y_max = max(y_max, 0.0)
    span = max(1e-6, y_max - y_min)
    pad = 0.12 * span
    txt_off = 0.02 * span

    ax.axhline(0.0, linewidth=1.0, alpha=0.6)

    for i, method_key in enumerate(ordered):
        vals = score[method_key].to_numpy(dtype=np.float64, copy=False)
        offset = (float(i) - float(n_methods) / 2.0) * width + width / 2.0
        xpos = x + offset

        finite = np.isfinite(vals)
        if finite.any():
            y = vals[finite]
            x_ok = xpos[finite]

            yerr = None
            if ci_lo is not None and ci_hi is not None and method_key in ci_lo.columns:
                lo = ci_lo[method_key].to_numpy(dtype=np.float64, copy=False)[finite]
                hi = ci_hi[method_key].to_numpy(dtype=np.float64, copy=False)[finite]
                lower = np.maximum(0.0, y - lo)
                upper = np.maximum(0.0, hi - y)
                yerr = np.vstack([lower, upper])

            highlight_alpha = (
                0.9 if (highlight_key is not None and method_key == highlight_key) else 0.75
            )

            ax.bar(
                x_ok,
                y,
                width,
                label=str(label_by_key.get(method_key, method_key)),
                color=colors[method_key],
                edgecolor="white",
                linewidth=0.5,
                alpha=highlight_alpha,
                yerr=yerr,
                capsize=2 if yerr is not None else 0,
            )

            nvals = (
                n_seeds[method_key].to_numpy(dtype=np.float64, copy=False)[finite]
                if method_key in n_seeds.columns
                else np.full_like(y, np.nan)
            )
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
                    fontsize=7,
                    alpha=0.9,
                )

        missing = ~finite
        if missing.any():
            for xx in xpos[missing]:
                ax.plot([float(xx)], [0.0], marker="x", markersize=4, alpha=0.6)
                ax.text(
                    float(xx),
                    float(0.0 + txt_off),
                    "∅",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    alpha=0.7,
                )

    ax.set_ylabel(f"Return vs baseline {baseline_label} (Δ / max(1, |baseline|))")
    ax.set_title("Baseline-relative performance (mean return, 95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(score.index.tolist(), rotation=30, ha="right")
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Method")

    fig.tight_layout()

    ensure_parent(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    fmt = out_path.suffix.lstrip(".").lower() or "png"
    fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format=fmt)
    atomic_replace(tmp, out_path)
    plt.close(fig)


def _as_str_scalar(x: object) -> str | None:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        return str(arr.reshape(-1)[0])
    except Exception:
        return None


def plot_trajectory_heatmap(npz_path: Path, out_path: Path, max_points: int = 20000) -> bool:
    if not npz_path.exists():
        return False

    try:
        data = np.load(npz_path, allow_pickle=True)
        obs = data["obs"]
        gates = data["gates"]
    except Exception:
        return False

    env_id = _as_str_scalar(data.get("env_id")) if hasattr(data, "get") else None
    method = _as_str_scalar(data.get("method")) if hasattr(data, "get") else None
    gate_source = _as_str_scalar(data.get("gate_source")) if hasattr(data, "get") else None

    env_disp = env_id or npz_path.stem.replace("_trajectory", "")
    proj = trajectory_projection(env_id, np.asarray(obs))
    if proj is None:
        return False

    xi, yi, xlab, ylab = proj
    obs_arr = np.asarray(obs)
    if obs_arr.ndim != 2 or obs_arr.shape[1] <= max(xi, yi):
        return False

    N = int(obs_arr.shape[0])
    if N > max_points:
        idx = np.linspace(0, N - 1, max_points, dtype=int)
        obs_arr = obs_arr[idx]
        gates = np.asarray(gates).reshape(-1)[idx]
    else:
        gates = np.asarray(gates).reshape(-1)

    x = obs_arr[:, int(xi)]
    y = obs_arr[:, int(yi)]

    g = np.asarray(gates).reshape(-1)
    if g.size != x.shape[0]:
        return False

    active = g >= 0.5
    gated = ~active

    gate_note = (gate_source or "recomputed").strip().lower()
    if gate_note not in {"checkpoint", "recomputed", "mixed", "n/a", "missing_intrinsic"}:
        gate_note = "recomputed"

    title_bits: list[str] = [str(env_disp)]
    if method:
        title_bits.append(str(method))
    title_bits.append(f"gates: {gate_note}")

    method_key = (str(method).strip().lower() if method is not None else "").strip() or "glpe"
    active_color = _color_for_method(method_key)

    fig, ax = plt.subplots(figsize=(8, 6))

    if gated.any():
        ax.scatter(
            x[gated],
            y[gated],
            c="lightgray",
            s=10,
            alpha=0.5,
            label="Gated/Off",
            edgecolor="none",
        )

    if active.any():
        ax.scatter(
            x[active],
            y[active],
            c=active_color,
            s=15,
            alpha=0.8,
            label="Active/On",
            edgecolor="none",
        )

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(" — ".join(title_bits))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ensure_parent(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    fmt = out_path.suffix.lstrip(".").lower() or "png"
    fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format=fmt)
    atomic_replace(tmp, out_path)
    plt.close(fig)
    return True
