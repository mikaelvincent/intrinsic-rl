from __future__ import annotations

from pathlib import Path
from typing import Sequence

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


_EVAL_SEMANTICS = "Evaluation (extrinsic return only)"


def _finite_minmax(vals: Sequence[float]) -> tuple[float, float] | None:
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


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    try:
        return float(np.trapezoid(y, x))
    except Exception:
        return float(np.trapz(y, x))


def _auc_from_curve(
    steps: np.ndarray,
    mean: np.ndarray,
    ci_lo: np.ndarray | None,
    ci_hi: np.ndarray | None,
) -> tuple[float, float | None, float | None, int]:
    x = np.asarray(steps, dtype=np.float64).reshape(-1)
    y = np.asarray(mean, dtype=np.float64).reshape(-1)

    if x.size == 0 or y.size == 0:
        return 0.0, None, None, 0

    n = int(min(x.size, y.size))
    x = x[:n]
    y = y[:n]

    lo = None
    hi = None
    if ci_lo is not None and ci_hi is not None:
        lo = np.asarray(ci_lo, dtype=np.float64).reshape(-1)
        hi = np.asarray(ci_hi, dtype=np.float64).reshape(-1)
        n2 = int(min(x.size, y.size, lo.size, hi.size))
        x = x[:n2]
        y = y[:n2]
        lo = lo[:n2]
        hi = hi[:n2]

    finite = np.isfinite(x) & np.isfinite(y)
    if not bool(finite.any()):
        return 0.0, None, None, 0

    x = x[finite]
    y = y[finite]
    if lo is not None and hi is not None:
        lo = lo[finite]
        hi = hi[finite]

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    if lo is not None and hi is not None:
        lo = lo[order]
        hi = hi[order]

    uniq: dict[int, tuple[float, float, float]] = {}
    for i in range(int(x.size)):
        s = int(x[i])
        y_i = float(y[i])
        if lo is None or hi is None:
            uniq[s] = (y_i, float("nan"), float("nan"))
        else:
            uniq[s] = (y_i, float(lo[i]), float(hi[i]))

    steps_i = np.asarray(sorted(uniq.keys()), dtype=np.float64)
    y_i = np.asarray([uniq[int(s)][0] for s in steps_i], dtype=np.float64)
    lo_i = np.asarray([uniq[int(s)][1] for s in steps_i], dtype=np.float64)
    hi_i = np.asarray([uniq[int(s)][2] for s in steps_i], dtype=np.float64)

    if steps_i.size == 0:
        return 0.0, None, None, 0

    if float(steps_i[0]) > 0.0:
        steps_i = np.concatenate([np.asarray([0.0], dtype=np.float64), steps_i])
        y_i = np.concatenate([np.asarray([y_i[0]], dtype=np.float64), y_i])
        lo_i = np.concatenate([np.asarray([lo_i[0]], dtype=np.float64), lo_i])
        hi_i = np.concatenate([np.asarray([hi_i[0]], dtype=np.float64), hi_i])

    auc = _trapezoid(y_i, steps_i)

    auc_lo = None
    auc_hi = None
    if np.isfinite(lo_i).any() and np.isfinite(hi_i).any():
        auc_lo = _trapezoid(lo_i, steps_i)
        auc_hi = _trapezoid(hi_i, steps_i)

    max_step = int(steps_i.max()) if steps_i.size else 0
    return float(auc), (None if auc_lo is None else float(auc_lo)), (None if auc_hi is None else float(auc_hi)), max_step


def plot_eval_auc_bars_by_env(
    by_step_df: pd.DataFrame,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
) -> list[Path]:
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

    label_by_key: dict[str, str] = {}
    if "method" in df.columns:
        label_by_key = (
            df.drop_duplicates(subset=["method_key"], keep="first")
            .set_index("method_key")["method"]
            .astype(str)
            .to_dict()
        )

    has_ci = {"mean_return_ci95_lo", "mean_return_ci95_hi"}.issubset(set(df.columns))

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    plt = _style()
    written: list[Path] = []

    for env_id in sorted(df["env_id"].unique().tolist()):
        df_env = df.loc[df["env_id"] == env_id].copy()
        if df_env.empty:
            continue

        auc_rows: list[dict[str, object]] = []
        for mk in want:
            df_m = df_env.loc[df_env["method_key"] == mk].copy()
            if df_m.empty:
                continue

            df_m = df_m.sort_values("ckpt_step").drop_duplicates(subset=["ckpt_step"], keep="last")

            steps = df_m["ckpt_step"].to_numpy(dtype=np.float64, copy=False)
            mean = pd.to_numeric(df_m["mean_return_mean"], errors="coerce").to_numpy(dtype=np.float64)

            if has_ci:
                lo = pd.to_numeric(df_m["mean_return_ci95_lo"], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                hi = pd.to_numeric(df_m["mean_return_ci95_hi"], errors="coerce").to_numpy(
                    dtype=np.float64
                )
            else:
                lo = None
                hi = None

            auc, auc_lo, auc_hi, max_step = _auc_from_curve(steps, mean, lo, hi)

            n_seeds = 0
            if "n_seeds" in df_m.columns:
                try:
                    n_seeds = int(pd.to_numeric(df_m["n_seeds"], errors="coerce").max())
                except Exception:
                    n_seeds = 0

            auc_rows.append(
                {
                    "method_key": mk,
                    "label": str(label_by_key.get(mk, mk)),
                    "auc": float(auc),
                    "auc_lo": None if auc_lo is None else float(auc_lo),
                    "auc_hi": None if auc_hi is None else float(auc_hi),
                    "n_seeds": int(n_seeds),
                    "max_step": int(max_step),
                }
            )

        if not auc_rows:
            continue

        labels = [str(r["label"]) for r in auc_rows]
        vals = np.asarray([float(r["auc"]) for r in auc_rows], dtype=np.float64)
        colors = [_color_for_method(str(r["method_key"])) for r in auc_rows]
        ns = [int(r.get("n_seeds", 0) or 0) for r in auc_rows]

        yerr_lo = np.zeros((len(auc_rows),), dtype=np.float64)
        yerr_hi = np.zeros((len(auc_rows),), dtype=np.float64)

        span_vals: list[float] = list(vals.tolist())

        for i, r in enumerate(auc_rows):
            lo_v = r.get("auc_lo")
            hi_v = r.get("auc_hi")
            if lo_v is None or hi_v is None:
                continue
            try:
                lo_f = float(lo_v)
                hi_f = float(hi_v)
            except Exception:
                continue
            if not (np.isfinite(lo_f) and np.isfinite(hi_f) and np.isfinite(vals[i])):
                continue

            lo_f, hi_f = (lo_f, hi_f) if lo_f <= hi_f else (hi_f, lo_f)
            yerr_lo[i] = max(0.0, float(vals[i] - lo_f))
            yerr_hi[i] = max(0.0, float(hi_f - vals[i]))
            span_vals.extend([lo_f, hi_f])

        mm = _finite_minmax(span_vals)
        y_lo, y_hi = (mm if mm is not None else (float(np.nanmin(vals)), float(np.nanmax(vals))))
        span = max(1e-9, float(y_hi - y_lo))
        txt_off = 0.02 * span

        x = np.arange(len(auc_rows), dtype=np.float64)

        fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(auc_rows)), 4.2))
        ax.bar(
            x,
            vals,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )

        if float(np.max(yerr_lo + yerr_hi)) > 0.0:
            ax.errorbar(
                x,
                vals,
                yerr=np.vstack([yerr_lo, yerr_hi]),
                fmt="none",
                ecolor="black",
                elinewidth=0.9,
                capsize=3,
                capthick=0.9,
                alpha=0.9,
            )

        for xi, yi, n in zip(x.tolist(), vals.tolist(), ns):
            ax.text(
                float(xi),
                float(yi + txt_off) if yi >= 0.0 else float(yi - txt_off),
                f"n={int(n)}" if int(n) > 0 else "n=?",
                ha="center",
                va="bottom" if yi >= 0.0 else "top",
                fontsize=8,
                alpha=0.9,
            )

        ax.axhline(0.0, linewidth=1.0, alpha=0.6, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel("Method")
        ax.set_ylabel(f"AUC (mean episode return × env steps) — {_EVAL_SEMANTICS}")
        ax.set_title(f"{env_id} — {title}", loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")

        if mm is not None:
            _set_y_minmax(ax, y_lo, y_hi)

        out = plots_root / f"{_env_tag(env_id)}__auc__{filename_suffix}.png"
        _save_fig(fig, out)
        plt.close(fig)
        written.append(out)

    return written
