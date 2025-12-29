from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import DPI, FIGSIZE, apply_grid


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

    ablation_mode = _is_ablation_suffix(filename_suffix)

    label_by_key = (
        df.drop_duplicates(subset=["method_key"], keep="first")
        .set_index("method_key")["method"]
        .astype(str)
        .to_dict()
        if "method" in df.columns
        else {}
    )

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    plt = apply_rcparams_paper()
    written: list[Path] = []

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

            auc, max_step = _auc_from_curve(steps, mean)

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

        x = np.arange(len(auc_rows), dtype=np.float64)

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=int(DPI))
        for i, (mk, v) in enumerate(zip([r["method_key"] for r in auc_rows], vals.tolist())):
            alpha = 1.0 if str(mk) == "glpe" else 0.88
            z = 10 if str(mk) == "glpe" else 2
            ax.bar(
                float(x[i]),
                float(v),
                color=colors[i],
                alpha=float(alpha),
                edgecolor="none",
                linewidth=0.0,
                zorder=z,
            )

        span = max(1e-9, float(np.nanmax(vals) - np.nanmin(vals))) if np.isfinite(vals).any() else 1.0
        txt_off = 0.02 * span
        for xi, yi, n in zip(x.tolist(), vals.tolist(), ns):
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
        ax.set_xlabel("Method")
        ax.set_ylabel("AUC (return × steps)")
        ax.set_title(f"{env_id} — {title}")

        apply_grid(ax)
        fig.tight_layout()

        out = plots_root / f"{_env_tag(env_id)}__auc__{filename_suffix}.png"
        save_fig_atomic(fig, out)
        plt.close(fig)
        written.append(out)

    return written
