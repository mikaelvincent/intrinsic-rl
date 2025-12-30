from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from irl.visualization.data import read_scalars
from irl.visualization.labels import add_row_label, env_label, method_label, slugify
from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import DPI, FIG_WIDTH, apply_grid


def _env_tag(env_id: str) -> str:
    return str(env_id).replace("/", "-")


def _is_ablation_suffix(filename_suffix: str) -> bool:
    return "ablation" in str(filename_suffix).strip().lower()


def _has_glpe_and_variant(method_keys: Sequence[str]) -> bool:
    keys = {str(k).strip().lower() for k in method_keys if str(k).strip()}
    if "glpe" not in keys:
        return False
    return any(k.startswith("glpe_") for k in keys)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    try:
        return float(np.trapezoid(y, x))
    except Exception:
        return float(np.trapz(y, x))


def _auc_from_curve(steps: np.ndarray, mean: np.ndarray) -> tuple[float, int]:
    x = np.asarray(steps, dtype=np.float64).reshape(-1)
    y = np.asarray(mean, dtype=np.float64).reshape(-1)

    if x.size == 0 or y.size == 0:
        return 0.0, 0

    n = int(min(x.size, y.size))
    x = x[:n]
    y = y[:n]

    finite = np.isfinite(x) & np.isfinite(y)
    if not bool(finite.any()):
        return 0.0, 0

    x = x[finite]
    y = y[finite]

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]

    uniq: dict[int, float] = {}
    for i in range(int(x.size)):
        uniq[int(x[i])] = float(y[i])

    steps_i = np.asarray(sorted(uniq.keys()), dtype=np.float64)
    y_i = np.asarray([uniq[int(s)] for s in steps_i], dtype=np.float64)

    if steps_i.size == 0:
        return 0.0, 0

    if float(steps_i[0]) > 0.0:
        steps_i = np.concatenate([np.asarray([0.0], dtype=np.float64), steps_i])
        y_i = np.concatenate([np.asarray([y_i[0]], dtype=np.float64), y_i])

    auc = _trapezoid(y_i, steps_i)
    max_step = int(steps_i.max()) if steps_i.size else 0
    return float(auc), int(max_step)


def paper_method_groups(methods: Sequence[str]) -> tuple[list[str], list[str]]:
    from irl.methods.spec import paper_method_groups as _paper_method_groups

    return _paper_method_groups(methods)


_TIME_COLS_UPDATE: tuple[str, ...] = (
    "time_rollout_s",
    "time_intrinsic_compute_s",
    "time_intrinsic_update_s",
    "time_gae_s",
    "time_ppo_s",
)


def _steps_per_update_from_config(run_dir: Path) -> int | None:
    cfg_path = Path(run_dir) / "config.json"
    try:
        text = cfg_path.read_text(encoding="utf-8")
        cfg = json.loads(text)
    except Exception:
        return None

    if not isinstance(cfg, dict):
        return None

    env = cfg.get("env")
    ppo = cfg.get("ppo")
    if not isinstance(env, dict) or not isinstance(ppo, dict):
        return None

    try:
        vec_envs = int(env.get("vec_envs", 1) or 1)
        rollout_steps = int(ppo.get("rollout_steps_per_env", 0) or 0)
    except Exception:
        return None

    vec_envs = max(1, int(vec_envs))
    rollout_steps = int(rollout_steps)
    if rollout_steps <= 0:
        return None

    return int(vec_envs) * int(rollout_steps)


def _time_curve_seconds(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        df = read_scalars(Path(run_dir))
    except Exception:
        return None

    if df.empty or "step" not in df.columns:
        return None

    df = df.loc[pd.to_numeric(df["step"], errors="coerce").notna()].copy()
    if df.empty:
        return None

    df["step"] = df["step"].astype(int)
    df = df.loc[df["step"] >= 0].copy()
    if df.empty:
        return None

    steps = df["step"].to_numpy(dtype=np.int64, copy=False)
    if int(steps.size) <= 1:
        step_out = np.asarray([0, int(steps[-1]) if int(steps.size) else 0], dtype=np.int64)
        time_out = np.asarray([0.0, 0.0], dtype=np.float64)
        return step_out, time_out

    dt = np.zeros((int(steps.size),), dtype=np.float64)
    for col in _TIME_COLS_UPDATE:
        if col not in df.columns:
            continue
        vals = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64, copy=False)
        )
        dt += vals

    dt = np.clip(dt, 0.0, None)

    steps_per_update = _steps_per_update_from_config(Path(run_dir))
    if steps_per_update is None:
        d = np.diff(steps.astype(np.int64, copy=False), axis=0)
        pos = d[d > 0]
        steps_per_update = int(np.median(pos)) if int(pos.size) else 0
    steps_per_update = int(max(1, int(steps_per_update)))

    steps_all = np.concatenate([np.asarray([0], dtype=np.int64), steps], axis=0)
    d_step = np.diff(steps_all).astype(np.int64, copy=False)

    updates = np.ceil(d_step.astype(np.float64) / float(steps_per_update))
    updates = np.where(np.isfinite(updates), updates, 1.0)
    updates_i = np.maximum(1, updates.astype(np.int64, copy=False))

    cum_s = np.cumsum(dt * updates_i, dtype=np.float64)

    step_out = np.concatenate([np.asarray([0], dtype=np.int64), steps], axis=0)
    time_out = np.concatenate([np.asarray([0.0], dtype=np.float64), cum_s], axis=0)
    return step_out, time_out


def _interp_extrap_1d(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    xs = np.asarray(x, dtype=np.float64).reshape(-1)
    ys = np.asarray(y, dtype=np.float64).reshape(-1)
    q = np.asarray(xq, dtype=np.float64).reshape(-1)

    if xs.size == 0 or ys.size == 0:
        return np.full((int(q.size),), np.nan, dtype=np.float64)

    n = int(min(xs.size, ys.size))
    xs = xs[:n]
    ys = ys[:n]

    order = np.argsort(xs, kind="mergesort")
    xs = xs[order]
    ys = ys[order]

    out = np.interp(q, xs, ys)

    if xs.size >= 2:
        dx = float(xs[-1] - xs[-2])
        if np.isfinite(dx) and dx > 0.0:
            slope = float(ys[-1] - ys[-2]) / dx
            hi = q > xs[-1]
            if bool(np.any(hi)):
                out[hi] = float(ys[-1]) + slope * (q[hi] - float(xs[-1]))

    return np.maximum(out, 0.0)


def _auc_from_time_curve(
    times_s: np.ndarray, returns: np.ndarray, *, budget_s: float
) -> tuple[float, float]:
    x = np.asarray(times_s, dtype=np.float64).reshape(-1)
    y = np.asarray(returns, dtype=np.float64).reshape(-1)

    if x.size == 0 or y.size == 0:
        return 0.0, 0.0

    n = int(min(x.size, y.size))
    x = x[:n]
    y = y[:n]

    finite = np.isfinite(x) & np.isfinite(y)
    if not bool(finite.any()):
        return 0.0, 0.0

    x = x[finite]
    y = y[finite]

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]

    uniq_x: list[float] = []
    uniq_y: list[float] = []
    for xi, yi in zip(x.tolist(), y.tolist()):
        if not uniq_x:
            uniq_x.append(float(xi))
            uniq_y.append(float(yi))
            continue
        if float(xi) == float(uniq_x[-1]):
            uniq_y[-1] = float(yi)
        else:
            uniq_x.append(float(xi))
            uniq_y.append(float(yi))

    xs = np.asarray(uniq_x, dtype=np.float64)
    ys = np.asarray(uniq_y, dtype=np.float64)
    if xs.size == 0:
        return 0.0, 0.0

    if float(xs[0]) > 0.0:
        xs = np.concatenate([np.asarray([0.0], dtype=np.float64), xs])
        ys = np.concatenate([np.asarray([ys[0]], dtype=np.float64), ys])

    budget = float(max(0.0, budget_s))
    if not np.isfinite(budget) or budget <= 0.0:
        return 0.0, 0.0

    if float(xs[-1]) < budget:
        xs = np.concatenate([xs, np.asarray([budget], dtype=np.float64)])
        ys = np.concatenate([ys, np.asarray([ys[-1]], dtype=np.float64)])
    elif float(xs[-1]) > budget:
        j = int(np.searchsorted(xs, budget, side="right") - 1)
        j = int(max(0, min(int(xs.size) - 1, j)))

        if float(xs[j]) == budget:
            xs = xs[: j + 1]
            ys = ys[: j + 1]
        else:
            yb = float(ys[j])
            if j + 1 < int(xs.size):
                x0 = float(xs[j])
                x1 = float(xs[j + 1])
                if np.isfinite(x1 - x0) and x1 > x0:
                    frac = float((budget - x0) / (x1 - x0))
                    yb = float(ys[j]) + frac * float(ys[j + 1] - ys[j])

            xs = np.concatenate([xs[: j + 1], np.asarray([budget], dtype=np.float64)])
            ys = np.concatenate([ys[: j + 1], np.asarray([yb], dtype=np.float64)])

    auc = _trapezoid(ys, xs)
    return float(auc), float(budget)


def plot_eval_auc_bars_by_env(
    by_step_df: pd.DataFrame,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
) -> list[Path]:
    _ = title
    if by_step_df is None or by_step_df.empty:
        return []

    required = {"env_id", "ckpt_step", "mean_return_mean"}
    if not required.issubset(set(by_step_df.columns)):
        return []

    df = by_step_df.copy()
    df["env_id"] = df["env_id"].astype(str).str.strip()

    if "method_key" not in df.columns:
        if "method" not in df.columns:
            return []
        df["method_key"] = df["method"].astype(str).str.strip().str.lower()

    df["method_key"] = df["method_key"].astype(str).str.strip().str.lower()

    df["ckpt_step"] = pd.to_numeric(df["ckpt_step"], errors="coerce")
    df = df.dropna(subset=["ckpt_step"]).copy()
    df["ckpt_step"] = df["ckpt_step"].astype(int)
    df = df.loc[df["ckpt_step"] >= 0].copy()

    want = [str(m).strip().lower() for m in methods_to_plot if str(m).strip()]
    if not want:
        return []

    ablation_mode = _is_ablation_suffix(filename_suffix)

    env_recs: list[tuple[str, list[dict[str, object]]]] = []

    for env_id in sorted(df["env_id"].unique().tolist()):
        df_env = df.loc[df["env_id"] == env_id].copy()
        if df_env.empty:
            continue

        methods_present = sorted(set(df_env["method_key"].unique().tolist()) & set(want))
        if not methods_present:
            continue

        if ablation_mode and not _has_glpe_and_variant(methods_present):
            continue

        auc_rows: list[dict[str, object]] = []
        for mk in want:
            df_m = df_env.loc[df_env["method_key"] == mk].copy()
            if df_m.empty:
                continue

            df_m = df_m.sort_values("ckpt_step").drop_duplicates(subset=["ckpt_step"], keep="last")

            steps = df_m["ckpt_step"].to_numpy(dtype=np.float64, copy=False)
            mean = pd.to_numeric(df_m["mean_return_mean"], errors="coerce").to_numpy(dtype=np.float64)

            auc, _max_step = _auc_from_curve(steps, mean)

            n_seeds = 0
            if "n_seeds" in df_m.columns:
                try:
                    n_seeds = int(pd.to_numeric(df_m["n_seeds"], errors="coerce").max())
                except Exception:
                    n_seeds = 0

            auc_rows.append(
                {
                    "method_key": mk,
                    "label": method_label(mk),
                    "auc": float(auc),
                    "n_seeds": int(n_seeds),
                }
            )

        if auc_rows:
            env_recs.append((str(env_id), auc_rows))

    if not env_recs:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    nrows = int(len(env_recs))
    height = max(2.8, 2.2 * float(nrows))

    plt = apply_rcparams_paper()
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(float(FIG_WIDTH), float(height)),
        dpi=int(DPI),
        squeeze=False,
    )

    for i, (env_id, auc_rows) in enumerate(env_recs):
        ax = axes[i, 0]

        labels = [str(r["label"]) for r in auc_rows]
        vals = np.asarray([float(r["auc"]) for r in auc_rows], dtype=np.float64)
        colors = [_color_for_method(str(r["method_key"])) for r in auc_rows]
        ns = [int(r.get("n_seeds", 0) or 0) for r in auc_rows]

        x = np.arange(len(auc_rows), dtype=np.float64)
        for j, r in enumerate(auc_rows):
            mk = str(r.get("method_key", "")).strip().lower()
            alpha = 1.0 if mk == "glpe" else 0.9
            z = 10 if mk == "glpe" else 2
            ax.bar(
                float(x[j]),
                float(vals[j]),
                color=colors[j],
                alpha=float(alpha),
                edgecolor="none",
                linewidth=0.0,
                zorder=z,
            )

        span = max(1e-9, float(np.nanmax(vals) - np.nanmin(vals))) if np.isfinite(vals).any() else 1.0
        txt_off = 0.02 * span
        for xi, yi, n in zip(x.tolist(), vals.tolist(), ns):
            if not np.isfinite(float(yi)):
                continue
            ax.text(
                float(xi),
                float(yi + txt_off) if yi >= 0.0 else float(yi - txt_off),
                f"n={int(n)}" if int(n) > 0 else "n=?",
                ha="center",
                va="bottom" if yi >= 0.0 else "top",
                fontsize=8,
                alpha=0.9,
                zorder=30,
            )

        ax.axhline(0.0, linewidth=1.0, alpha=0.6, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("AUC (return × steps)")
        apply_grid(ax)

        if i != nrows - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)

        add_row_label(ax, env_label(env_id))

    axes[-1, 0].set_xlabel("Method")
    fig.tight_layout()

    out = Path(plots_root) / f"eval-auc-steps-{slugify(filename_suffix)}.png"
    save_fig_atomic(fig, out)
    plt.close(fig)
    return [out]


def plot_eval_auc_time_bars_by_env(
    by_step_df: pd.DataFrame,
    *,
    timing_groups: Mapping[str, Mapping[str, Sequence[Path]]],
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
) -> list[Path]:
    _ = title
    if by_step_df is None or by_step_df.empty:
        return []

    required = {"env_id", "ckpt_step", "mean_return_mean"}
    if not required.issubset(set(by_step_df.columns)):
        return []

    df = by_step_df.copy()
    df["env_id"] = df["env_id"].astype(str).str.strip()

    if "method_key" not in df.columns:
        if "method" not in df.columns:
            return []
        df["method_key"] = df["method"].astype(str).str.strip().str.lower()

    df["method_key"] = df["method_key"].astype(str).str.strip().str.lower()

    df["ckpt_step"] = pd.to_numeric(df["ckpt_step"], errors="coerce")
    df = df.dropna(subset=["ckpt_step"]).copy()
    df["ckpt_step"] = df["ckpt_step"].astype(int)
    df = df.loc[df["ckpt_step"] >= 0].copy()

    want = [str(m).strip().lower() for m in methods_to_plot if str(m).strip()]
    if not want:
        return []

    ablation_mode = _is_ablation_suffix(filename_suffix)

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    curve_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def _mean_times(env_id: str, method_key: str, ckpt_steps: np.ndarray) -> np.ndarray | None:
        env_key = _env_tag(env_id)
        by_method = timing_groups.get(env_key)
        if not isinstance(by_method, Mapping):
            return None
        run_dirs = by_method.get(str(method_key))
        if not isinstance(run_dirs, (list, tuple)) or not run_dirs:
            return None

        per_run: list[np.ndarray] = []
        q = np.asarray(ckpt_steps, dtype=np.float64).reshape(-1)

        for rd in run_dirs:
            rp = Path(rd)
            try:
                cache_key = str(rp.resolve())
            except Exception:
                cache_key = str(rp)

            if cache_key in curve_cache:
                curve = curve_cache[cache_key]
            else:
                curve = _time_curve_seconds(rp)
                curve_cache[cache_key] = curve

            if curve is None:
                continue

            s, t = curve
            per_run.append(_interp_extrap_1d(s, t, q))

        if not per_run:
            return None

        arr = np.stack(per_run, axis=0).astype(np.float64, copy=False)
        mean = np.mean(arr, axis=0)
        return mean.astype(np.float64, copy=False)

    env_recs: list[tuple[str, list[dict[str, object]], float]] = []

    for env_id in sorted(df["env_id"].unique().tolist()):
        df_env = df.loc[df["env_id"] == env_id].copy()
        if df_env.empty:
            continue

        methods_present = sorted(set(df_env["method_key"].unique().tolist()) & set(want))
        if not methods_present:
            continue

        if ablation_mode and not _has_glpe_and_variant(methods_present):
            continue

        curves: list[dict[str, object]] = []

        for mk in want:
            if mk not in set(methods_present):
                continue

            df_m = df_env.loc[df_env["method_key"] == mk].copy()
            if df_m.empty:
                continue

            df_m = df_m.sort_values("ckpt_step").drop_duplicates(subset=["ckpt_step"], keep="last")

            steps = df_m["ckpt_step"].to_numpy(dtype=np.int64, copy=False)
            returns = pd.to_numeric(df_m["mean_return_mean"], errors="coerce").to_numpy(dtype=np.float64)

            ok = np.isfinite(steps.astype(np.float64)) & np.isfinite(returns)
            steps = steps[ok]
            returns = returns[ok]
            if int(steps.size) == 0:
                continue

            times = _mean_times(env_id, mk, steps)
            if times is None:
                continue

            ok2 = np.isfinite(times) & np.isfinite(returns)
            if not bool(np.any(ok2)):
                continue

            steps = steps[ok2]
            returns = returns[ok2]
            times = np.asarray(times, dtype=np.float64).reshape(-1)[ok2]
            if int(times.size) == 0:
                continue

            n_seeds = 0
            if "n_seeds" in df_m.columns:
                try:
                    n_seeds = int(pd.to_numeric(df_m["n_seeds"], errors="coerce").max())
                except Exception:
                    n_seeds = 0

            curves.append(
                {
                    "method_key": mk,
                    "label": method_label(mk),
                    "times_s": times.astype(np.float64, copy=False),
                    "returns": returns.astype(np.float64, copy=False),
                    "n_seeds": int(n_seeds),
                    "t_end_s": float(times[-1]),
                }
            )

        if not curves:
            continue

        t_ends = [
            float(c["t_end_s"])
            for c in curves
            if np.isfinite(float(c["t_end_s"])) and float(c["t_end_s"]) > 0.0
        ]
        if not t_ends:
            continue

        budget_s = float(min(t_ends))
        if not (np.isfinite(budget_s) and budget_s > 0.0):
            continue

        auc_rows: list[dict[str, object]] = []
        for c in curves:
            auc, _tmax = _auc_from_time_curve(
                np.asarray(c["times_s"], dtype=np.float64),
                np.asarray(c["returns"], dtype=np.float64),
                budget_s=budget_s,
            )
            auc_rows.append(
                {
                    "method_key": str(c["method_key"]),
                    "label": str(c["label"]),
                    "auc": float(auc),
                    "n_seeds": int(c["n_seeds"]),
                }
            )

        if auc_rows:
            env_recs.append((str(env_id), auc_rows, float(budget_s)))

    if not env_recs:
        return []

    nrows = int(len(env_recs))
    height = max(2.8, 2.2 * float(nrows))

    plt = apply_rcparams_paper()
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(float(FIG_WIDTH), float(height)),
        dpi=int(DPI),
        squeeze=False,
    )

    for i, (env_id, auc_rows, budget_s) in enumerate(env_recs):
        ax = axes[i, 0]

        labels = [str(r["label"]) for r in auc_rows]
        vals = np.asarray([float(r["auc"]) for r in auc_rows], dtype=np.float64)
        colors = [_color_for_method(str(r["method_key"])) for r in auc_rows]
        ns = [int(r.get("n_seeds", 0) or 0) for r in auc_rows]

        x = np.arange(len(auc_rows), dtype=np.float64)

        for j, r in enumerate(auc_rows):
            mk = str(r.get("method_key", "")).strip().lower()
            alpha = 1.0 if mk == "glpe" else 0.9
            z = 10 if mk == "glpe" else 2
            ax.bar(
                float(x[j]),
                float(vals[j]),
                color=colors[j],
                alpha=float(alpha),
                edgecolor="none",
                linewidth=0.0,
                zorder=z,
            )

        span = max(1e-9, float(np.nanmax(vals) - np.nanmin(vals))) if np.isfinite(vals).any() else 1.0
        txt_off = 0.02 * span
        for xi, yi, n in zip(x.tolist(), vals.tolist(), ns):
            if not np.isfinite(float(yi)):
                continue
            ax.text(
                float(xi),
                float(yi + txt_off) if yi >= 0.0 else float(yi - txt_off),
                f"n={int(n)}" if int(n) > 0 else "n=?",
                ha="center",
                va="bottom" if yi >= 0.0 else "top",
                fontsize=8,
                alpha=0.9,
                zorder=30,
            )

        ax.axhline(0.0, linewidth=1.0, alpha=0.6, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("AUC (return × time)")
        apply_grid(ax)

        if i != nrows - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)

        budget_min = float(budget_s) / 60.0
        add_row_label(ax, f"{env_label(env_id)} ({budget_min:.1f} min)")

    axes[-1, 0].set_xlabel("Method")
    fig.tight_layout()

    out = Path(plots_root) / f"eval-auc-time-{slugify(filename_suffix)}.png"
    save_fig_atomic(fig, out)
    plt.close(fig)
    return [out]
