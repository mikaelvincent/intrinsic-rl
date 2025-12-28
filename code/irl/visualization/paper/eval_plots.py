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

    plt = _style()
    written: list[Path] = []

    for env_id in sorted(raw_df["env_id"].unique().tolist()):
        df_env = raw_df.loc[raw_df["env_id"] == env_id].copy()
        if df_env.empty:
            continue

        methods_present = sorted(set(df_env["method_key"].tolist()))
        methods = methods_present if want is None else [m for m in want if m in set(methods_present)]
        if not methods:
            continue

        uniq_steps = pd.to_numeric(df_env["ckpt_step"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        uniq_steps = uniq_steps[np.isfinite(uniq_steps)]
        uniq_steps = np.unique(uniq_steps)

        def _x_jitter_scale(steps: np.ndarray, n: int) -> float:
            if int(n) <= 1:
                return 0.0
            s = np.asarray(steps, dtype=np.float64).reshape(-1)
            s = s[np.isfinite(s)]
            if int(s.size) < 2:
                return 0.0
            diffs = np.diff(np.sort(s))
            diffs = diffs[diffs > 0.0]
            if int(diffs.size) == 0:
                return 0.0
            return float(np.min(diffs)) * 0.02

        x_jitter = _x_jitter_scale(uniq_steps, len(methods))

        fig, ax = plt.subplots(figsize=(8.8, 4.8))

        x_all: list[float] = []
        y_all: list[float] = []

        for i, mk in enumerate(methods):
            df_m = df_env.loc[df_env["method_key"] == mk].copy()
            if df_m.empty:
                continue

            x = pd.to_numeric(df_m["ckpt_step"], errors="coerce").to_numpy(dtype=np.float64)
            y = pd.to_numeric(df_m["mean_return"], errors="coerce").to_numpy(dtype=np.float64)

            finite = np.isfinite(x) & np.isfinite(y)
            if not bool(finite.any()):
                continue

            x = x[finite]
            y = y[finite]

            if x_jitter != 0.0:
                off = (float(i) - (float(len(methods)) - 1.0) / 2.0) * float(x_jitter)
                x = x + off

            ax.scatter(
                x,
                y,
                s=float(point_size),
                alpha=float(alpha),
                color=_color_for_method(mk),
                edgecolor="none",
                label=str(label_by_key.get(mk, mk)),
            )

            x_all.extend([float(v) for v in x.tolist()])
            y_all.extend([float(v) for v in y.tolist()])

        if not x_all or not y_all:
            plt.close(fig)
            continue

        ax.set_xlabel("Checkpoint step (env steps)")
        ax.set_ylabel("Mean episode return (eval)")
        ax.set_title(f"{env_id} — {title}", loc="left", fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")

        x_mm = _finite_minmax(x_all)
        if x_mm is not None:
            if float(x_mm[0]) == float(x_mm[1]):
                pad = 1.0
                ax.set_xlim(float(x_mm[0]) - pad, float(x_mm[1]) + pad)
            else:
                pad = 0.06 * float(x_mm[1] - x_mm[0])
                ax.set_xlim(float(x_mm[0]) - pad, float(x_mm[1]) + pad)

        y_mm = _finite_minmax(y_all)
        if y_mm is not None:
            _set_y_minmax(ax, y_mm[0], y_mm[1])

        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, title="Method")

        note = "points=per seed×checkpoint (summary_raw.csv)"
        if x_jitter != 0.0:
            note += f" | x_jitter≈{x_jitter:.3g}"
        fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8, alpha=0.9)

        fig.tight_layout(rect=[0.0, 0.04, 0.82, 1.0])

        out = plots_root / f"{_env_tag(env_id)}__eval_scatter__{filename_suffix}.png"
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


def _half_threshold(threshold: float) -> float:
    thr = float(threshold)
    if not np.isfinite(thr):
        return thr
    if thr >= 0.0:
        return 0.5 * thr
    return 1.5 * thr


def _fmt_threshold(v: float) -> str:
    if not np.isfinite(float(v)):
        return "nan"
    if abs(float(v) - round(float(v))) < 1e-9:
        return str(int(round(float(v))))
    return f"{float(v):.3g}"


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


def plot_steps_to_beat_by_env(
    by_step_df: pd.DataFrame,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
    summary_raw_csv: Path | None = None,
) -> Path | None:
    if by_step_df.empty:
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

    methods_present = sorted(set(raw_df["method_key"].unique().tolist()))
    methods = [m for m in want if m in set(methods_present)]
    if not methods:
        return None

    envs_present = sorted(set(raw_df["env_id"].unique().tolist()))
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
    threshold_sources: dict[str, str] = {}
    for env_id in envs:
        df_env = by_step_df.loc[by_step_df["env_id"].astype(str).str.strip() == str(env_id)].copy()
        thr, src = _score_to_beat(str(env_id), df_env)
        thresholds_full[str(env_id)] = float(thr)
        threshold_sources[str(env_id)] = str(src)

    thresholds_half = {e: _half_threshold(float(thresholds_full[e])) for e in envs}

    plt = _style()
    import matplotlib.patches as mpatches

    n_env = int(len(envs))
    n_methods = int(len(methods))
    width = 0.8 / float(max(1, n_methods))
    x = np.arange(n_env, dtype=np.float64)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(8.0, 1.6 + 1.15 * float(n_env)), 7.4),
        sharex=True,
    )

    env_tick_labels = [
        f"{e}\nthr={_fmt_threshold(float(thresholds_full.get(e, float('nan'))))}" for e in envs
    ]

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
        if bool(finite.any()):
            ymin = float(np.nanmin(q25s[finite]))
            ymax = float(np.nanmax(q75s[finite]))
            if not np.isfinite(ymin) or ymin <= 0.0:
                ymin = float(np.nanmin(meds[finite]))
            ymin = max(1.0, float(ymin))
            ymax = max(ymin, float(ymax))
        else:
            max_step = float(raw_df.loc[raw_df["env_id"].isin(envs), "ckpt_step"].max())
            ymax = max(1.0, float(max_step) if np.isfinite(max_step) else 1.0)
            ymin = 1.0

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
            if ok.any():
                ax.bar(
                    xpos[ok],
                    y[ok],
                    width=width,
                    color=_color_for_method(mk),
                    alpha=0.85 if mk == "glpe" else 0.75,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=2,
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
                    zorder=3,
                )

        ax.set_ylabel("Steps to reach")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)

        if use_log:
            ax.set_yscale("log")
            ax.set_ylim(ymin * 0.8, ymax * 1.35)
            base_y = ymin * 1.05
            text_mult = 1.08
            y_off = 0.0
        else:
            ax.set_ylim(0.0, ymax * 1.15)
            base_y = 0.0 + 0.02 * ymax
            text_mult = 1.0
            y_off = 0.03 * ymax

        for ei in range(n_env):
            for mi in range(n_methods):
                tot = int(n_total[ei, mi])
                got = int(n_reached[ei, mi])

                off = (float(mi) - float(n_methods) / 2.0) * width + width / 2.0
                xpos = float(x[ei] + off)

                if tot <= 0:
                    ax.text(xpos, base_y, "∅", ha="center", va="bottom", fontsize=7, alpha=0.75)
                    continue

                if got <= 0 or not np.isfinite(meds[ei, mi]):
                    ax.plot([xpos], [base_y], marker="x", markersize=4, alpha=0.65, color="black")
                    ax.text(
                        xpos,
                        base_y,
                        f"0/{tot}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        alpha=0.85,
                    )
                    continue

                yv = float(meds[ei, mi])
                y_text = yv * text_mult if use_log else yv + y_off
                ax.text(
                    xpos,
                    y_text,
                    f"{got}/{tot}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    alpha=0.9,
                )

        if not bool(finite.any()):
            ax.text(
                0.5,
                0.55,
                "No runs reached this threshold",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                alpha=0.85,
            )

    _panel(axes[0], thresholds=thresholds_full, label="Solved threshold (successful runs only)")
    _panel(axes[1], thresholds=thresholds_half, label="Half threshold (successful runs only)")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(env_tick_labels, rotation=20, ha="right")
    axes[-1].set_xlabel("Environment")

    handles = [
        mpatches.Patch(color=_color_for_method(m), label=str(label_by_key.get(m, m))) for m in methods
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        title="Method",
    )

    thr_bits = [
        f"{e}={_fmt_threshold(float(thresholds_full.get(e, float('nan'))))}({threshold_sources.get(e,'')})"
        for e in envs
    ]
    note = (
        "Bars=median(IQR) steps over reaching runs; labels=reached/total; "
        "non-reaching runs excluded. "
        "Full thresholds: " + "; ".join(thr_bits) + ". "
        "Half thresholds: 0.5× full for ≥0, 1.5× full for <0."
    )
    fig.suptitle(title, y=0.995, fontweight="bold")
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8, alpha=0.9)

    fig.tight_layout(rect=[0.0, 0.05, 0.84, 0.97])

    out = Path(plots_root) / f"steps_to_beat__{filename_suffix}__success_only__full_and_half.png"
    _save_fig(fig, out)
    plt.close(fig)
    return out
