from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic, sort_env_ids as _sort_env_ids
from irl.visualization.style import DPI, LEGEND_FRAMEALPHA, LEGEND_FONTSIZE, apply_grid
from .thresholds import _SUPPORTED_SCORE_ENVS, fmt_threshold, solved_threshold

_SOLVED_MIN_REACH_FRAC: float = 0.25
_SOLVED_MIN_REACH_COUNT: int = 2


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


def _is_ablation_suffix(filename_suffix: str) -> bool:
    return "ablation" in str(filename_suffix).strip().lower()


def _eligible_ablation_envs(raw_df: pd.DataFrame) -> set[str]:
    eligible: set[str] = set()
    if raw_df is None or raw_df.empty:
        return eligible

    for env_id, g in raw_df.groupby("env_id", sort=False):
        methods = {str(m).strip().lower() for m in g["method_key"].unique().tolist()}
        if "glpe" not in methods:
            continue
        if any(m.startswith("glpe_") for m in methods):
            eligible.add(str(env_id))
    return eligible


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


def _best_returns_from_summary_raw(df_env_raw: pd.DataFrame) -> np.ndarray:
    if df_env_raw is None or df_env_raw.empty:
        return np.empty((0,), dtype=np.float64)

    cols = set(df_env_raw.columns.tolist())
    if not {"method_key", "seed", "mean_return"}.issubset(cols):
        return np.empty((0,), dtype=np.float64)

    df = df_env_raw[["method_key", "seed", "mean_return"]].copy()
    df["mean_return"] = pd.to_numeric(df["mean_return"], errors="coerce")
    df = df.dropna(subset=["mean_return"]).copy()
    if df.empty:
        return np.empty((0,), dtype=np.float64)

    try:
        best = df.groupby(["method_key", "seed"], sort=False)["mean_return"].max()
    except Exception:
        return np.empty((0,), dtype=np.float64)

    vals = pd.to_numeric(best, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    vals = vals[np.isfinite(vals)]
    return vals


def _maybe_lower_solved_threshold(
    thr_ref: float,
    best_returns: np.ndarray,
    *,
    min_reach_frac: float,
    min_reach_count: int,
) -> float:
    if best_returns.size == 0 or not np.isfinite(float(thr_ref)):
        return float(thr_ref)

    n = int(best_returns.size)
    reach = int(np.sum(best_returns >= float(thr_ref)))
    required_count = int(min(int(min_reach_count), n))

    required_frac = float(max(float(min_reach_frac), float(required_count) / float(n))) if n > 0 else 1.0
    reach_frac = float(reach) / float(n) if n > 0 else 0.0

    if reach >= required_count and reach_frac >= required_frac:
        return float(thr_ref)

    q = float(np.clip(1.0 - required_frac, 0.0, 1.0))
    thr_q = float(np.quantile(best_returns, q))
    thr_out = float(min(float(thr_ref), float(thr_q)))
    return float(thr_ref) if not np.isfinite(thr_out) else thr_out


def _score_to_beat(
    env_id: str,
    df_env: pd.DataFrame,
    *,
    df_env_raw: pd.DataFrame | None = None,
    min_reach_frac: float = _SOLVED_MIN_REACH_FRAC,
    min_reach_count: int = _SOLVED_MIN_REACH_COUNT,
) -> float:
    thr = solved_threshold(str(env_id))
    if thr is None:
        best = _best_final_threshold(df_env)
        thr = 0.0 if best is None else float(best)

    if df_env_raw is not None:
        best_returns = _best_returns_from_summary_raw(df_env_raw)
        thr = _maybe_lower_solved_threshold(
            float(thr),
            best_returns,
            min_reach_frac=float(min_reach_frac),
            min_reach_count=int(min_reach_count),
        )

    return float(thr)


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


def _half_threshold(threshold: float) -> float:
    thr = float(threshold)
    if not np.isfinite(thr):
        return thr
    if thr >= 0.0:
        return 0.5 * thr
    return 1.5 * thr


def plot_steps_to_beat_by_env(
    by_step_df: pd.DataFrame,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
    summary_raw_csv: Path | None = None,
) -> Path | None:
    if by_step_df is None or by_step_df.empty:
        return None

    required = {"env_id", "method_key", "ckpt_step", "mean_return_mean"}
    if not required.issubset(set(by_step_df.columns)):
        return None

    if summary_raw_csv is None or not Path(summary_raw_csv).exists():
        return None

    raw_df = _load_summary_raw(Path(summary_raw_csv))
    if raw_df is None or raw_df.empty:
        return None

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    want = [str(m).strip().lower() for m in methods_to_plot if str(m).strip()]
    if not want:
        return None

    ablation_mode = _is_ablation_suffix(filename_suffix)

    label_by_key = (
        by_step_df.drop_duplicates(subset=["method_key"], keep="first")
        .set_index("method_key")["method"]
        .to_dict()
        if "method" in by_step_df.columns
        else {}
    )

    raw_df = raw_df.loc[raw_df["method_key"].isin(want)].copy()
    raw_df = raw_df.loc[raw_df["env_id"].isin(_SUPPORTED_SCORE_ENVS)].copy()
    if raw_df.empty:
        return None

    if ablation_mode:
        eligible_envs = _eligible_ablation_envs(raw_df)
        if not eligible_envs:
            return None
        raw_df = raw_df.loc[raw_df["env_id"].isin(eligible_envs)].copy()
        if raw_df.empty:
            return None

    methods_present = sorted(set(raw_df["method_key"].unique().tolist()))
    methods = [m for m in want if m in set(methods_present)]
    if not methods:
        return None

    envs_present = _sort_env_ids(raw_df["env_id"].unique().tolist())
    envs = [e for e in envs_present if e in _SUPPORTED_SCORE_ENVS]
    if not envs:
        return None

    envs = [e for e in envs if not raw_df.loc[raw_df["env_id"] == e].empty]
    if not envs:
        return None

    by_env_method: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray] | None]] = {}
    for (env_id, method_key, seed), g in raw_df.groupby(["env_id", "method_key", "seed"], sort=False):
        df_g = g.sort_values("ckpt_step").drop_duplicates(subset=["ckpt_step"], keep="last")
        steps = df_g["ckpt_step"].to_numpy(dtype=np.float64, copy=False)
        vals = df_g["mean_return"].to_numpy(dtype=np.float64, copy=False)

        ok = np.isfinite(steps) & np.isfinite(vals)
        steps = steps[ok]
        vals = vals[ok]

        curve = None if steps.size == 0 else (steps, vals)
        by_env_method.setdefault((str(env_id), str(method_key)), []).append(curve)

    thresholds_full: dict[str, float] = {}
    for env_id in envs:
        df_env = by_step_df.loc[by_step_df["env_id"].astype(str).str.strip() == str(env_id)].copy()
        df_env_raw = raw_df.loc[raw_df["env_id"].astype(str).str.strip() == str(env_id)].copy()
        thresholds_full[str(env_id)] = float(_score_to_beat(str(env_id), df_env, df_env_raw=df_env_raw))

    thresholds_half = {e: _half_threshold(float(thresholds_full[e])) for e in envs}

    plt = apply_rcparams_paper()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(9.0, 1.1 + 1.0 * float(len(envs))), 7.4),
        dpi=int(DPI),
        sharex=True,
    )

    n_env = int(len(envs))
    n_methods = int(len(methods))
    width = 0.8 / float(max(1, n_methods))
    x = np.arange(n_env, dtype=np.float64)

    def _panel(ax, *, thresholds: Mapping[str, float], label: str) -> None:
        meds = np.full((n_env, n_methods), np.nan, dtype=np.float64)
        q25s = np.full((n_env, n_methods), np.nan, dtype=np.float64)
        q75s = np.full((n_env, n_methods), np.nan, dtype=np.float64)
        n_reached = np.zeros((n_env, n_methods), dtype=np.int64)
        n_total = np.zeros((n_env, n_methods), dtype=np.int64)

        for ei, env_id in enumerate(envs):
            thr = float(thresholds.get(env_id, float("nan")))
            for mi, mk in enumerate(methods):
                curves = by_env_method.get((env_id, mk), [])
                n_total[ei, mi] = int(len(curves))
                hits: list[float] = []
                for c in curves:
                    if c is None:
                        continue
                    s, v = c
                    hit = _steps_to_reach_threshold(s, v, thr)
                    if hit is None or not np.isfinite(float(hit)):
                        continue
                    hits.append(float(hit))

                n_reached[ei, mi] = int(len(hits))
                if not hits:
                    continue

                arr = np.asarray(hits, dtype=np.float64)
                q25, med, q75 = np.quantile(arr, [0.25, 0.5, 0.75])
                q25s[ei, mi] = float(q25)
                meds[ei, mi] = float(med)
                q75s[ei, mi] = float(q75)

        finite = np.isfinite(meds)
        use_log = False
        if bool(finite.any()):
            finite_pos = meds[finite] > 0.0
            if bool(finite_pos.any()):
                mn = float(np.nanmin(meds[finite][finite_pos]))
                mx = float(np.nanmax(meds[finite][finite_pos]))
                if mn > 0.0 and mx / mn >= 50.0:
                    use_log = True

        for mi, mk in enumerate(methods):
            off = (float(mi) - float(n_methods) / 2.0) * width + width / 2.0
            xpos = x + off
            y = meds[:, mi]
            ok = np.isfinite(y)
            if not bool(ok.any()):
                continue

            alpha = 1.0 if str(mk).strip().lower() == "glpe" else 0.88
            z = 10 if str(mk).strip().lower() == "glpe" else 2

            ax.bar(
                xpos[ok],
                y[ok],
                width=width,
                color=_color_for_method(mk),
                alpha=float(alpha),
                edgecolor="none",
                linewidth=0.0,
                zorder=z,
            )

            lo = np.maximum(0.0, y[ok] - q25s[:, mi][ok])
            hi = np.maximum(0.0, q75s[:, mi][ok] - y[ok])
            ax.errorbar(
                xpos[ok],
                y[ok],
                yerr=np.vstack([lo, hi]),
                fmt="none",
                ecolor="black",
                elinewidth=0.9,
                capsize=2,
                capthick=0.9,
                alpha=0.9,
                zorder=30,
            )

        ax.set_ylabel("Steps to reach")
        ax.set_title(label)
        apply_grid(ax)

        if use_log:
            ax.set_yscale("log")

        def _pct_label(got: int, tot: int) -> str:
            if int(tot) <= 0:
                return "âˆ…"
            pct = int(round(100.0 * float(got) / float(tot)))
            pct = int(max(0, min(100, pct)))
            return f"{pct}%"

        ymax = float(np.nanmax(q75s[finite])) if bool(finite.any()) else float(
            raw_df.loc[raw_df["env_id"].isin(envs), "ckpt_step"].max()
        )
        ymax = float(ymax) if np.isfinite(ymax) else 1.0
        base_y = 1.05 if use_log else 0.0 + 0.03 * ymax

        for ei in range(n_env):
            for mi in range(n_methods):
                tot = int(n_total[ei, mi])
                got = int(n_reached[ei, mi])
                off = (float(mi) - float(n_methods) / 2.0) * width + width / 2.0
                xpos = float(x[ei] + off)
                txt = _pct_label(got, tot)

                if tot <= 0:
                    ax.text(xpos, base_y, txt, ha="center", va="bottom", fontsize=6, alpha=0.75)
                    continue

                if got <= 0 or not np.isfinite(meds[ei, mi]):
                    ax.text(xpos, base_y, txt, ha="center", va="bottom", fontsize=6, alpha=0.9)
                    continue

                yv = float(meds[ei, mi])
                y_text = yv * 1.08 if use_log else yv + 0.03 * ymax
                ax.text(xpos, y_text, txt, ha="center", va="bottom", fontsize=6, alpha=0.9)

        thr_labels = []
        for env_id in envs:
            thr_labels.append(
                f"{env_id}: {fmt_threshold(float(thresholds.get(env_id, float('nan'))))}"
            )
        _ = thr_labels

    _panel(axes[0], thresholds=thresholds_full, label="Solved threshold (successful runs only)")
    _panel(axes[1], thresholds=thresholds_half, label="Half threshold (successful runs only)")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(e) for e in envs], rotation=20, ha="right")
    axes[-1].set_xlabel("Environment")

    handles = []
    labels = []
    for m in methods:
        handles.append(
            plt.Line2D(
                [],
                [],
                color=_color_for_method(m),
                lw=3.0 if m == "glpe" else 2.0,
            )
        )
        labels.append(str(label_by_key.get(m, m)))

    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower right",
        framealpha=float(LEGEND_FRAMEALPHA),
        fontsize=int(LEGEND_FONTSIZE),
        title=None,
    )

    fig.suptitle(str(title))
    fig.tight_layout()

    out = (
        Path(plots_root)
        / f"steps_to_beat__{filename_suffix}__success_only__full_and_half.png"
    )
    save_fig_atomic(fig, out)
    plt.close(fig)
    return out
