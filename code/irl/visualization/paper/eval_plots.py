from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..palette import color_for_method as _color_for_method
from ..plot_utils import apply_rcparams_paper, save_fig_atomic


def _style():
    return apply_rcparams_paper()


def _save_fig(fig, path: Path) -> None:
    save_fig_atomic(fig, Path(path))


def _env_tag(env_id: str) -> str:
    return str(env_id).replace("/", "-")


def _finite_minmax(vals: Iterable[float]) -> tuple[float, float] | None:
    arr = np.asarray([float(v) for v in vals], dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.min()), float(arr.max())


def _set_y_minmax(ax, lo: float, hi: float) -> None:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return
    if float(lo) == float(hi):
        pad = 1.0 if abs(float(lo)) < 1.0 else 0.05 * abs(float(lo))
        ax.set_ylim(float(lo) - pad, float(hi) + pad)
        return
    span = float(hi - lo)
    pad = max(1e-6, 0.08 * span)
    ax.set_ylim(float(lo) - pad, float(hi) + pad)


def plot_eval_bars_by_env(
    summary_df: pd.DataFrame,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
) -> list[Path]:
    if summary_df.empty:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    want = [str(m).strip().lower() for m in methods_to_plot if str(m).strip()]
    if not want:
        return []

    label_by_key = (
        summary_df.drop_duplicates(subset=["method_key"], keep="first")
        .set_index("method_key")["method"]
        .to_dict()
    )

    written: list[Path] = []
    plt = _style()

    for env_id in sorted(summary_df["env_id"].unique().tolist()):
        df_env = summary_df.loc[summary_df["env_id"] == env_id].copy()
        if df_env.empty:
            continue

        rows_by_method: dict[str, Mapping[str, Any]] = {}
        for _, r in df_env.iterrows():
            mk = str(r.get("method_key", "")).strip().lower()
            if mk:
                rows_by_method[mk] = r

        methods_present = [m for m in want if m in rows_by_method]
        if not methods_present:
            continue

        means = np.asarray(
            [float(rows_by_method[m]["mean_return_mean"]) for m in methods_present], dtype=np.float64
        )
        ci_lo = np.asarray(
            [float(rows_by_method[m]["mean_return_ci95_lo"]) for m in methods_present],
            dtype=np.float64,
        )
        ci_hi = np.asarray(
            [float(rows_by_method[m]["mean_return_ci95_hi"]) for m in methods_present],
            dtype=np.float64,
        )
        n_seeds = [int(rows_by_method[m].get("n_seeds", 0) or 0) for m in methods_present]

        y_minmax = _finite_minmax(list(ci_lo) + list(ci_hi) + list(means))
        if y_minmax is None:
            continue
        y_lo, y_hi = y_minmax

        labels = [str(label_by_key.get(m, m)) for m in methods_present]
        colors = [_color_for_method(m) for m in methods_present]

        x = np.arange(len(methods_present), dtype=np.float64)

        fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(methods_present)), 4.2))
        ax.bar(
            x,
            means,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )

        yerr = np.vstack([np.maximum(0.0, means - ci_lo), np.maximum(0.0, ci_hi - means)])
        ax.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=3,
            capthick=0.9,
            alpha=0.9,
        )

        for xi, yi, n in zip(x.tolist(), means.tolist(), n_seeds):
            ax.text(
                float(xi),
                float(yi),
                f"n={int(n)}",
                ha="center",
                va="bottom" if yi >= 0.0 else "top",
                fontsize=8,
                alpha=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel("Method")
        ax.set_ylabel("Mean episode return (eval)")
        ax.set_title(f"{env_id} — {title}", loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        _set_y_minmax(ax, y_lo, y_hi)

        out = plots_root / f"{_env_tag(env_id)}__{filename_suffix}.png"
        _save_fig(fig, out)
        plt.close(fig)
        written.append(out)

    return written


def plot_eval_curves_by_env(
    by_step_df: pd.DataFrame,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
) -> list[Path]:
    if by_step_df.empty:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    want = [str(m).strip().lower() for m in methods_to_plot if str(m).strip()]
    if not want:
        return []

    label_by_key = (
        by_step_df.drop_duplicates(subset=["method_key"], keep="first")
        .set_index("method_key")["method"]
        .to_dict()
    )

    written: list[Path] = []
    plt = _style()

    for env_id in sorted(by_step_df["env_id"].unique().tolist()):
        df_env = by_step_df.loc[by_step_df["env_id"] == env_id].copy()
        if df_env.empty:
            continue

        methods_present = sorted(set(df_env["method_key"].tolist()) & set(want))
        if not methods_present:
            continue

        uniq_steps = sorted(set(df_env["ckpt_step"].tolist()))
        if len(uniq_steps) <= 1:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 4.6))

        all_ci_lo: list[float] = []
        all_ci_hi: list[float] = []

        for mk in want:
            if mk not in set(methods_present):
                continue

            df_m = df_env.loc[df_env["method_key"] == mk].copy()
            if df_m.empty:
                continue
            df_m = df_m.sort_values("ckpt_step")

            x = df_m["ckpt_step"].to_numpy(dtype=np.int64, copy=False)
            y = df_m["mean_return_mean"].to_numpy(dtype=np.float64, copy=False)
            lo = df_m["mean_return_ci95_lo"].to_numpy(dtype=np.float64, copy=False)
            hi = df_m["mean_return_ci95_hi"].to_numpy(dtype=np.float64, copy=False)

            all_ci_lo.extend(lo.tolist())
            all_ci_hi.extend(hi.tolist())

            c = _color_for_method(mk)
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=str(label_by_key.get(mk, mk)),
                color=c,
                alpha=0.95,
            )
            ax.fill_between(
                x,
                lo,
                hi,
                color=c,
                alpha=0.12,
                linewidth=0.0,
            )

        y_minmax = _finite_minmax(all_ci_lo + all_ci_hi)
        if y_minmax is not None:
            _set_y_minmax(ax, y_minmax[0], y_minmax[1])

        ax.set_xlabel("Checkpoint step (env steps)")
        ax.set_ylabel("Mean episode return (eval)")
        ax.set_title(f"{env_id} — {title}", loc="left", fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(loc="best")

        out = plots_root / f"{_env_tag(env_id)}__{filename_suffix}.png"
        _save_fig(fig, out)
        plt.close(fig)
        written.append(out)

    return written


_SUPPORTED_SCORE_ENVS: set[str] = {
    "Ant-v5",
    "BipedalWalker-v3",
    "CarRacing-v3",
    "HalfCheetah-v5",
    "Humanoid-v5",
    "MountainCar-v0",
}


_KNOWN_SCORE_THRESHOLDS: dict[str, float] = {
    "Ant-v5": 6000.0,
    "BipedalWalker-v3": 300.0,
    "CarRacing-v3": 900.0,
    "HalfCheetah-v5": 4800.0,
    "Humanoid-v5": 6000.0,  # paper-based (Gymnasium spec uses None)
    "MountainCar-v0": -110.0,
}


def _reward_threshold_from_gym_spec(env_id: str) -> float | None:
    try:
        import gymnasium as gym  # type: ignore
    except Exception:
        return None

    try:
        spec = gym.spec(str(env_id))
    except Exception:
        return None

    rt = getattr(spec, "reward_threshold", None)
    if rt is None:
        return None

    try:
        v = float(rt)
    except Exception:
        return None

    return v if np.isfinite(v) else None


def _reward_threshold_from_known(env_id: str) -> float | None:
    key = str(env_id).strip()
    if key in _KNOWN_SCORE_THRESHOLDS:
        return float(_KNOWN_SCORE_THRESHOLDS[key])
    return None


def _best_final_threshold(df_env: pd.DataFrame) -> float | None:
    if df_env.empty:
        return None
    if "ckpt_step" not in df_env.columns or "mean_return_mean" not in df_env.columns:
        return None

    steps = pd.to_numeric(df_env["ckpt_step"], errors="coerce")
    if steps.isna().all():
        return None
    max_step = int(steps.max())

    df_final = df_env.loc[df_env["ckpt_step"] == max_step]
    if df_final.empty:
        df_final = df_env

    vals = pd.to_numeric(df_final["mean_return_mean"], errors="coerce").to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(vals.max())


def _score_to_beat(env_id: str, df_env: pd.DataFrame) -> tuple[float, str]:
    rt = _reward_threshold_from_gym_spec(env_id)
    if rt is not None:
        return float(rt), "gym_spec"

    rt = _reward_threshold_from_known(env_id)
    if rt is not None:
        return float(rt), "known"

    rt = _best_final_threshold(df_env)
    if rt is not None:
        return float(rt), "best_final"

    return 0.0, "fallback"


def _steps_to_reach_threshold(steps: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    s = np.asarray(steps, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if s.size == 0 or v.size == 0 or s.size != v.size:
        return None
    if not np.isfinite(float(threshold)):
        return None

    for i in range(int(s.size)):
        vi = float(v[i])
        if not np.isfinite(vi):
            continue
        if vi < float(threshold):
            continue

        if i == 0:
            return float(s[i])

        v0 = float(v[i - 1])
        s0 = float(s[i - 1])
        s1 = float(s[i])

        if not np.isfinite(v0):
            return float(s1)

        denom = float(vi - v0)
        if denom <= 0.0 or not np.isfinite(denom):
            return float(s1)

        frac = float((float(threshold) - v0) / denom)
        if not np.isfinite(frac):
            return float(s1)
        frac = float(np.clip(frac, 0.0, 1.0))
        return float(s0 + frac * (s1 - s0))

    return None


def plot_steps_to_beat_by_env(
    by_step_df: pd.DataFrame,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
) -> Path | None:
    if by_step_df.empty:
        return None

    required = {"env_id", "method_key", "ckpt_step", "mean_return_mean"}
    if not required.issubset(set(by_step_df.columns)):
        return None

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    want = [str(m).strip().lower() for m in methods_to_plot if str(m).strip()]
    if not want:
        return None

    label_by_key = (
        by_step_df.drop_duplicates(subset=["method_key"], keep="first")
        .set_index("method_key")["method"]
        .to_dict()
        if "method" in by_step_df.columns
        else {}
    )

    envs = sorted(by_step_df["env_id"].astype(str).str.strip().unique().tolist())
    if not envs:
        return None

    envs = [e for e in envs if e in _SUPPORTED_SCORE_ENVS]
    if not envs:
        return None

    methods_present = sorted(set(by_step_df["method_key"].astype(str).str.strip().str.lower().tolist()))
    methods = [m for m in want if m in set(methods_present)]
    if not methods:
        return None

    per_env_threshold: dict[str, float] = {}
    per_env_source: dict[str, str] = {}
    per_env_max_step: dict[str, int] = {}

    for env_id in envs:
        df_env = by_step_df.loc[by_step_df["env_id"].astype(str).str.strip() == str(env_id)].copy()
        if df_env.empty:
            continue

        try:
            per_env_max_step[env_id] = int(pd.to_numeric(df_env["ckpt_step"], errors="coerce").max())
        except Exception:
            per_env_max_step[env_id] = 0

        thr, src = _score_to_beat(str(env_id), df_env)
        per_env_threshold[env_id] = float(thr)
        per_env_source[env_id] = str(src)

    max_step_global = 0
    if per_env_max_step:
        max_step_global = int(max(per_env_max_step.values()))

    vals = np.full((len(envs), len(methods)), np.nan, dtype=np.float64)
    censored = np.zeros((len(envs), len(methods)), dtype=bool)

    for e_i, env_id in enumerate(envs):
        df_env = by_step_df.loc[by_step_df["env_id"].astype(str).str.strip() == str(env_id)].copy()
        if df_env.empty:
            continue

        thr = float(per_env_threshold.get(env_id, float("nan")))
        max_step_env = int(per_env_max_step.get(env_id, max_step_global) or 0)
        censor_height = float(max_step_env) if max_step_env > 0 else float(max(1, max_step_global))

        for m_i, mk in enumerate(methods):
            df_m = df_env.loc[
                df_env["method_key"].astype(str).str.strip().str.lower() == str(mk)
            ].copy()
            if df_m.empty:
                censored[e_i, m_i] = True
                vals[e_i, m_i] = 1.05 * float(censor_height)
                continue

            df_m = df_m.sort_values("ckpt_step")
            steps = pd.to_numeric(df_m["ckpt_step"], errors="coerce").to_numpy(dtype=np.float64)
            scores = pd.to_numeric(df_m["mean_return_mean"], errors="coerce").to_numpy(dtype=np.float64)

            ok = np.isfinite(steps) & np.isfinite(scores)
            steps = steps[ok]
            scores = scores[ok]
            if steps.size == 0:
                censored[e_i, m_i] = True
                vals[e_i, m_i] = 1.05 * float(censor_height)
                continue

            hit = _steps_to_reach_threshold(steps, scores, float(thr))
            if hit is None:
                censored[e_i, m_i] = True
                vals[e_i, m_i] = 1.05 * float(censor_height)
            else:
                vals[e_i, m_i] = float(hit)

    plt = _style()
    n_env = int(len(envs))
    n_methods = int(len(methods))
    width = 0.8 / float(max(1, n_methods))
    x = np.arange(n_env, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 + 1.15 * float(n_env)), 5.0))

    for i, mk in enumerate(methods):
        y = vals[:, i].astype(np.float64, copy=False)
        off = (float(i) - float(n_methods) / 2.0) * width + width / 2.0
        xpos = x + off

        color = _color_for_method(mk)
        label = str(label_by_key.get(mk, mk))

        bars = ax.bar(
            xpos,
            y,
            width=width,
            color=color,
            alpha=0.88 if mk == "glpe" else 0.75,
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )

        for b, is_cens in zip(bars, censored[:, i].tolist()):
            if bool(is_cens):
                b.set_hatch("//")
                b.set_alpha(0.35)

    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in envs], rotation=20, ha="right")
    ax.set_ylabel("Env steps to reach score threshold")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    finite_pos = vals[np.isfinite(vals) & (vals > 0.0)]
    if finite_pos.size >= 2:
        ratio = float(finite_pos.max() / max(1e-12, float(finite_pos.min())))
        if ratio >= 50.0:
            ax.set_yscale("log")

    fig.text(
        0.01,
        0.01,
        "Hatched = not reached by final checkpoint",
        ha="left",
        va="bottom",
        fontsize=8,
        alpha=0.9,
    )

    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, title="Method")
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 1.0])

    out = plots_root / f"steps_to_beat__{filename_suffix}.png"
    _save_fig(fig, out)
    plt.close(fig)
    return out
