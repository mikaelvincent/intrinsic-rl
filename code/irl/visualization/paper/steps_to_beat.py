from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from irl.visualization.labels import add_legend_rows_top, add_row_label, env_label, legend_ncol, method_label, slugify
from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic, sort_env_ids as _sort_env_ids
from irl.visualization.style import (
    DPI,
    FIG_WIDTH,
    LEGEND_FRAMEALPHA,
    LEGEND_FONTSIZE,
    apply_grid,
    legend_order as _legend_order,
)
from .thresholds import _SUPPORTED_SCORE_ENVS, solved_threshold

_SOLVED_MIN_REACH_FRAC: float = 0.25
_SOLVED_MIN_REACH_COUNT: int = 2

_BAR_ALPHA_MIN: float = 0.25
_BAR_ALPHA_MAX: float = 0.95
_LABEL_TEXT_PAD_FRAC: float = 0.03
_LABEL_BG_ALPHA: float = 0.25


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

    required_frac = (
        float(max(float(min_reach_frac), float(required_count) / float(n))) if n > 0 else 1.0
    )
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


def _threshold_fraction(threshold: float, frac: float) -> float:
    thr = float(threshold)
    f = float(frac)
    if not (np.isfinite(thr) and np.isfinite(f)):
        return thr
    if f >= 1.0:
        return thr
    if f <= 0.0:
        return thr
    if thr >= 0.0:
        return thr * f
    return thr * (2.0 - f)


def _reach_label(reached: int, total: int) -> str:
    tot = int(total)
    if tot <= 0:
        return "â€”"
    r = int(reached)
    r = int(max(0, min(r, tot)))
    return f"{r}/{tot}"


def _reach_to_alpha(reached: int, total: int) -> float:
    if int(total) <= 0:
        p = 0.0
    else:
        p = float(int(reached)) / float(int(total))
    p = float(np.clip(p, 0.0, 1.0))
    return float(_BAR_ALPHA_MIN + (_BAR_ALPHA_MAX - _BAR_ALPHA_MIN) * p)


def _y_text_for_patch(patch: object, *, use_log: bool, pad_frac: float) -> float:
    try:
        y0 = float(getattr(patch, "get_y")())
        h = float(getattr(patch, "get_height")())
    except Exception:
        return float("nan")

    if not np.isfinite(y0) or not np.isfinite(h):
        return float("nan")

    y1 = y0 + h
    if not np.isfinite(y1):
        return float("nan")

    frac = float(np.clip(float(pad_frac), 0.0, 0.5))

    if not use_log:
        return float(y0 + frac * h)

    if y0 <= 0.0 or y1 <= 0.0:
        return float(y0 + frac * h)

    ratio = float(y1 / y0)
    if not np.isfinite(ratio) or ratio <= 1.0:
        return float(y0)

    return float(y0 * (ratio**frac))


def plot_steps_to_beat_by_env(
    by_step_df: pd.DataFrame,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
    summary_raw_csv: Path | None = None,
) -> Path | None:
    _ = title
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

    want = _legend_order(methods_to_plot)
    if not want:
        return None

    ablation_mode = _is_ablation_suffix(filename_suffix)

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

    thresholds_half = {e: _threshold_fraction(float(thresholds_full[e]), 0.5) for e in envs}
    thresholds_quarter = {e: _threshold_fraction(float(thresholds_full[e]), 0.25) for e in envs}

    plt = apply_rcparams_paper()

    thr_dicts: tuple[Mapping[str, float], ...] = (thresholds_full, thresholds_half, thresholds_quarter)
    thr_labels: tuple[str, ...] = ("100%", "50%", "25%")

    n_env = int(len(envs))
    n_thr = int(len(thr_dicts))
    n_methods = int(len(methods))

    width = 0.8 / float(max(1, n_methods))
    x = np.arange(n_thr, dtype=np.float64)

    fig_h = max(3.4, 2.3 * float(n_env))
    fig, axes = plt.subplots(
        n_env,
        1,
        figsize=(float(FIG_WIDTH), float(fig_h)),
        dpi=int(DPI),
        sharex=True,
        squeeze=False,
    )

    def _panel(ax, *, env_id: str) -> None:
        import matplotlib.colors as mcolors

        meds = np.full((n_thr, n_methods), np.nan, dtype=np.float64)
        stds = np.full((n_thr, n_methods), np.nan, dtype=np.float64)
        n_reached = np.zeros((n_thr, n_methods), dtype=np.int64)
        n_total = np.zeros((n_thr, n_methods), dtype=np.int64)

        for ti, thr_map in enumerate(thr_dicts):
            thr = float(thr_map.get(str(env_id), float("nan")))
            for mi, mk in enumerate(methods):
                curves = by_env_method.get((str(env_id), str(mk)), [])
                n_total[ti, mi] = int(len(curves))

                hits: list[float] = []
                for c in curves:
                    if c is None:
                        continue
                    s, v = c
                    hit = _steps_to_reach_threshold(s, v, thr)
                    if hit is None or not np.isfinite(float(hit)):
                        continue
                    hits.append(float(hit))

                n_reached[ti, mi] = int(len(hits))
                if not hits:
                    continue

                arr = np.asarray(hits, dtype=np.float64)
                meds[ti, mi] = float(np.median(arr))
                stds[ti, mi] = float(np.std(arr, ddof=0))

        finite_pos = meds[np.isfinite(meds) & (meds > 0.0)]
        use_log = False
        if finite_pos.size >= 2:
            ratio = float(finite_pos.max() / finite_pos.min())
            if np.isfinite(ratio) and ratio >= 50.0:
                use_log = True

        base = 0.0
        if use_log and finite_pos.size > 0:
            base = float(max(1.0, 0.01 * float(finite_pos.min())))

        base_floor = float(max(0.0, base))

        patch_by_idx: dict[tuple[int, int], object] = {}

        for mi, mk in enumerate(methods):
            off = (float(mi) - float(n_methods) / 2.0) * width + width / 2.0
            xpos = x + off

            y = meds[:, mi]
            ok = np.isfinite(y) & (y > base if use_log else np.isfinite(y))
            if not bool(ok.any()):
                continue

            y_top = y[ok]
            heights = (y_top - base) if use_log else y_top

            bars = ax.bar(
                xpos[ok],
                heights,
                width=width,
                bottom=float(base),
                color=_color_for_method(mk),
                edgecolor="black",
                linewidth=0.9,
                zorder=10 if str(mk).strip().lower() == "glpe" else 2,
            )

            base_rgba = mcolors.to_rgba(_color_for_method(mk))
            thr_idxs = np.flatnonzero(ok).astype(int, copy=False)
            for patch, ti in zip(bars.patches, thr_idxs.tolist()):
                a = _reach_to_alpha(int(n_reached[ti, mi]), int(n_total[ti, mi]))
                patch.set_facecolor((base_rgba[0], base_rgba[1], base_rgba[2], float(a)))
                patch.set_edgecolor((0.0, 0.0, 0.0, 1.0))
                patch.set_linewidth(0.9)
                patch_by_idx[(int(ti), int(mi))] = patch

            err = stds[:, mi][ok]
            err = np.where(np.isfinite(err) & (err >= 0.0), err, 0.0)

            lower = np.minimum(err, y_top - base_floor)
            lower = np.clip(lower, 0.0, None)

            ax.errorbar(
                xpos[ok],
                y_top,
                yerr=np.vstack([lower, err]),
                fmt="none",
                ecolor="black",
                elinewidth=0.9,
                capsize=2,
                capthick=0.9,
                alpha=0.9,
                zorder=30,
            )

        bbox = {
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": float(_LABEL_BG_ALPHA),
        }

        xform = ax.get_xaxis_transform()

        for ti in range(n_thr):
            for mi in range(n_methods):
                reached = int(n_reached[ti, mi])
                total = int(n_total[ti, mi])
                txt = _reach_label(reached, total)

                patch = patch_by_idx.get((int(ti), int(mi)))
                if patch is None:
                    if total > 0 and reached == 0:
                        off = (float(mi) - float(n_methods) / 2.0) * width + width / 2.0
                        cx = float(x[ti] + off)
                        ax.text(
                            float(cx),
                            0.02,
                            txt,
                            transform=xform,
                            ha="center",
                            va="bottom",
                            rotation=90,
                            fontsize=8,
                            color="black",
                            bbox=bbox,
                            clip_on=True,
                            zorder=40,
                        )
                    continue

                try:
                    cx = float(patch.get_x()) + 0.5 * float(patch.get_width())
                except Exception:
                    continue

                y_text = _y_text_for_patch(
                    patch,
                    use_log=bool(use_log),
                    pad_frac=float(_LABEL_TEXT_PAD_FRAC),
                )
                if not np.isfinite(float(y_text)):
                    continue

                ax.text(
                    float(cx),
                    float(y_text),
                    txt,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=8,
                    color="black",
                    bbox=bbox,
                    clip_on=True,
                    clip_path=patch,
                    zorder=40,
                )

        ax.set_xlim(-0.5, float(n_thr) - 0.5)
        ax.set_ylabel("Training steps")
        ax.set_xticks(x)

        apply_grid(ax)
        add_row_label(ax, env_label(env_id))

        if use_log:
            ax.set_yscale("log")

    for i, env_id in enumerate(envs):
        ax = axes[i, 0]
        _panel(ax, env_id=str(env_id))
        if i != n_env - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    axes[-1, 0].set_xticklabels(list(thr_labels))
    axes[-1, 0].set_xlabel("Threshold fraction")

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
        labels.append(method_label(m))

    top = 1.0
    if handles:
        top = add_legend_rows_top(fig, [(handles, labels, legend_ncol(len(handles)))])

    fig.tight_layout(rect=[0.0, 0.0, 1.0, float(top)])

    out = Path(plots_root) / f"steps-to-beat-{slugify(filename_suffix)}.png"
    save_fig_atomic(fig, out)
    plt.close(fig)
    return out
