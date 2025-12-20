from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from irl.utils.checkpoint import atomic_replace


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    return plt


def _save_fig(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fmt = path.suffix.lstrip(".").lower() or "png"
    fig.savefig(str(tmp), dpi=300, bbox_inches="tight", format=fmt)
    atomic_replace(tmp, path)


def _env_tag(env_id: str) -> str:
    return str(env_id).replace("/", "-")


_BASELINE_ORDER: tuple[str, ...] = ("vanilla", "icm", "rnd", "ride", "riac", "glpe")
_ABLATION_ORDER: tuple[str, ...] = ("glpe_lp_only", "glpe_impact_only", "glpe_nogate", "glpe_cache", "glpe")


def paper_method_groups(methods: Sequence[str]) -> tuple[list[str], list[str]]:
    ms = [str(m).strip().lower() for m in methods if str(m).strip()]
    ms_set = set(ms)

    baselines: list[str] = [m for m in _BASELINE_ORDER if m in ms_set]
    extras = sorted([m for m in ms if m not in set(baselines) and not m.startswith("glpe_") and m != "glpe"])
    baselines.extend([m for m in extras if m not in set(baselines)])

    ablations: list[str] = [m for m in _ABLATION_ORDER if m in ms_set]
    other_abls = sorted([m for m in ms if m.startswith("glpe_") and m not in set(ablations)])
    ablations.extend([m for m in other_abls if m not in set(ablations)])

    if "glpe" in ms_set:
        if "glpe" not in baselines:
            baselines.append("glpe")
        if "glpe" not in ablations:
            ablations.append("glpe")

    return baselines, ablations


def load_eval_summary_table(summary_csv: Path) -> pd.DataFrame:
    p = Path(summary_csv)
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Empty summary table: {p}")

    required = {
        "method",
        "env_id",
        "mean_return_mean",
        "mean_return_ci95_lo",
        "mean_return_ci95_hi",
        "n_seeds",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"summary.csv missing columns {missing}: {p}")

    out = df.copy()
    out["env_id"] = out["env_id"].astype(str).str.strip()
    out["method"] = out["method"].astype(str).str.strip()
    out["method_key"] = out["method"].str.lower().str.strip()
    return out


def load_eval_by_step_table(summary_by_step_csv: Path) -> pd.DataFrame:
    p = Path(summary_by_step_csv)
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Empty by-step summary table: {p}")

    required = {
        "method",
        "env_id",
        "ckpt_step",
        "mean_return_mean",
        "mean_return_ci95_lo",
        "mean_return_ci95_hi",
        "n_seeds",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"summary_by_step.csv missing columns {missing}: {p}")

    out = df.copy()
    out["env_id"] = out["env_id"].astype(str).str.strip()
    out["method"] = out["method"].astype(str).str.strip()
    out["method_key"] = out["method"].str.lower().str.strip()
    out["ckpt_step"] = pd.to_numeric(out["ckpt_step"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["ckpt_step"]).copy()
    out["ckpt_step"] = out["ckpt_step"].astype(int)
    return out


def _color_for_method(method_key: str) -> str:
    mk = str(method_key).strip().lower()
    if mk == "vanilla":
        return "#7f7f7f"
    if mk == "glpe":
        return "#d62728"
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#bcbd22",
        "#17becf",
    ]
    idx = abs(hash(mk)) % len(palette)
    return palette[idx]


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
            [float(rows_by_method[m]["mean_return_ci95_lo"]) for m in methods_present], dtype=np.float64
        )
        ci_hi = np.asarray(
            [float(rows_by_method[m]["mean_return_ci95_hi"]) for m in methods_present], dtype=np.float64
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

            ax.plot(
                x,
                y,
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=str(label_by_key.get(mk, mk)),
                color=_color_for_method(mk),
                alpha=0.95,
            )
            ax.fill_between(
                x,
                lo,
                hi,
                color=_color_for_method(mk),
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


def _npz_str(data: Any, key: str) -> str | None:
    try:
        arr = np.asarray(data[key])
        if arr.size == 0:
            return None
        return str(arr.reshape(-1)[0])
    except Exception:
        return None


def _trajectory_projection(env_id: str | None, obs: np.ndarray) -> tuple[int, int, str, str] | None:
    if obs.ndim != 2 or obs.shape[1] < 2:
        return None

    D = int(obs.shape[1])
    e = (env_id or "").strip()

    if e.startswith("MountainCar"):
        return 0, 1, "position", "velocity"
    if e.startswith("CartPole"):
        return (0, 2, "cart_pos", "pole_angle") if D >= 3 else (0, 1, "obs[0]", "obs[1]")
    if e.startswith("Pendulum"):
        return 0, 1, "cos(theta)", "sin(theta)"
    if e.startswith("Acrobot"):
        return 0, 1, "cos(theta1)", "sin(theta1)"
    if e.startswith("LunarLander"):
        return 0, 1, "x", "y"
    if e.startswith("BipedalWalker") and D >= 2:
        return 0, 1, "obs[0]", "obs[1]"

    if D == 2:
        return 0, 1, "obs[0]", "obs[1]"

    return None


def _sample_idx(n: int, k: int, *, seed: int) -> np.ndarray:
    nn = int(n)
    kk = int(k)
    if kk <= 0 or nn <= kk:
        return np.arange(nn, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return rng.choice(nn, size=kk, replace=False).astype(np.int64, copy=False)


def plot_glpe_state_gate_map(
    *,
    traj_root: Path,
    plots_root: Path,
    max_points: int = 40000,
) -> list[Path]:
    traj_root = Path(traj_root)
    if not traj_root.exists():
        return []

    by_env: dict[str, list[Path]] = {}
    for p in sorted(traj_root.rglob("*_trajectory.npz"), key=lambda x: str(x)):
        try:
            data = np.load(p, allow_pickle=False)
        except Exception:
            continue

        method = _npz_str(data, "method")
        env_id = _npz_str(data, "env_id")
        if method is None or env_id is None:
            continue
        if str(method).strip().lower() != "glpe":
            continue

        by_env.setdefault(str(env_id), []).append(p)

    if not by_env:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)
    plt = _style()

    written: list[Path] = []

    for env_id, files in sorted(by_env.items(), key=lambda kv: str(kv[0])):
        obs_all: list[np.ndarray] = []
        gates_all: list[np.ndarray] = []

        for p in files:
            try:
                data = np.load(p, allow_pickle=False)
                obs = np.asarray(data["obs"], dtype=np.float32)
                gates = np.asarray(data["gates"]).reshape(-1)
            except Exception:
                continue
            if obs.ndim != 2 or gates.size != obs.shape[0]:
                continue
            obs_all.append(obs)
            gates_all.append(gates.astype(np.float32, copy=False))

        if not obs_all:
            continue

        obs_cat = np.concatenate(obs_all, axis=0)
        gates_cat = np.concatenate(gates_all, axis=0)
        proj = _trajectory_projection(env_id, obs_cat)
        if proj is None:
            continue

        xi, yi, xlab, ylab = proj
        x = obs_cat[:, int(xi)]
        y = obs_cat[:, int(yi)]
        g = gates_cat >= 0.5

        idx = _sample_idx(x.shape[0], int(max_points), seed=abs(hash(env_id)) & 0x7FFFFFFF)
        x = x[idx]
        y = y[idx]
        g = g[idx]

        fig, ax = plt.subplots(figsize=(7.8, 6.0))

        if (~g).any():
            ax.scatter(
                x[~g],
                y[~g],
                c="lightgray",
                s=8,
                alpha=0.35,
                label="Gated/Off",
                edgecolor="none",
            )
        if g.any():
            ax.scatter(
                x[g],
                y[g],
                c="#d62728",
                s=10,
                alpha=0.75,
                label="Active/On",
                edgecolor="none",
            )

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"{env_id} — GLPE gate map (eval)", loc="left", fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25, linestyle="--")

        x_mm = _finite_minmax(x.tolist())
        y_mm = _finite_minmax(y.tolist())
        if x_mm is not None and y_mm is not None:
            if x_mm[0] != x_mm[1]:
                pad = 0.06 * (x_mm[1] - x_mm[0])
                ax.set_xlim(x_mm[0] - pad, x_mm[1] + pad)
            if y_mm[0] != y_mm[1]:
                pad = 0.06 * (y_mm[1] - y_mm[0])
                ax.set_ylim(y_mm[0] - pad, y_mm[1] + pad)

        out = plots_root / f"{_env_tag(env_id)}__glpe_gate_map.png"
        _save_fig(fig, out)
        plt.close(fig)
        written.append(out)

    return written


def plot_glpe_extrinsic_vs_intrinsic(
    *,
    traj_root: Path,
    plots_root: Path,
    max_points: int = 60000,
) -> list[Path]:
    traj_root = Path(traj_root)
    if not traj_root.exists():
        return []

    by_env: dict[str, list[Path]] = {}
    for p in sorted(traj_root.rglob("*_trajectory.npz"), key=lambda x: str(x)):
        try:
            data = np.load(p, allow_pickle=False)
        except Exception:
            continue

        method = _npz_str(data, "method")
        env_id = _npz_str(data, "env_id")
        if method is None or env_id is None:
            continue
        if str(method).strip().lower() != "glpe":
            continue
        if "rewards_ext" not in getattr(data, "files", []):
            continue

        by_env.setdefault(str(env_id), []).append(p)

    if not by_env:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)
    plt = _style()

    written: list[Path] = []

    for env_id, files in sorted(by_env.items(), key=lambda kv: str(kv[0])):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        gs: list[np.ndarray] = []

        for p in files:
            try:
                data = np.load(p, allow_pickle=False)
                r_ext = np.asarray(data["rewards_ext"], dtype=np.float32).reshape(-1)
                r_int = np.asarray(data["intrinsic"], dtype=np.float32).reshape(-1)
                gates = np.asarray(data["gates"]).reshape(-1)
            except Exception:
                continue

            n = int(min(r_ext.size, r_int.size, gates.size))
            if n <= 0:
                continue

            xs.append(r_ext[:n])
            ys.append(r_int[:n])
            gs.append(gates[:n].astype(np.float32, copy=False))

        if not xs:
            continue

        x = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
        y = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
        g = np.concatenate(gs, axis=0) >= 0.5

        idx = _sample_idx(x.shape[0], int(max_points), seed=abs(hash(("extint", env_id))) & 0x7FFFFFFF)
        x = x[idx]
        y = y[idx]
        g = g[idx]

        x_mm = _finite_minmax(x.tolist())
        y_mm = _finite_minmax(y.tolist())
        if x_mm is None or y_mm is None:
            continue

        fig, ax = plt.subplots(figsize=(6.8, 5.6))

        if (~g).any():
            ax.scatter(
                x[~g],
                y[~g],
                s=10,
                alpha=0.35,
                c="lightgray",
                edgecolor="none",
                label="Gated/Off",
            )
        if g.any():
            ax.scatter(
                x[g],
                y[g],
                s=12,
                alpha=0.65,
                c="#d62728",
                edgecolor="none",
                label="Active/On",
            )

        ax.axhline(0.0, linewidth=1.0, alpha=0.4, color="black")
        ax.axvline(0.0, linewidth=1.0, alpha=0.4, color="black")

        ax.set_xlabel("Extrinsic reward (per step)")
        ax.set_ylabel("Intrinsic reward (per step)")
        ax.set_title(f"{env_id} — GLPE extrinsic vs intrinsic (eval)", loc="left", fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(loc="best")

        if x_mm[0] != x_mm[1]:
            pad = 0.08 * (x_mm[1] - x_mm[0])
            ax.set_xlim(x_mm[0] - pad, x_mm[1] + pad)
        if y_mm[0] != y_mm[1]:
            pad = 0.08 * (y_mm[1] - y_mm[0])
            ax.set_ylim(y_mm[0] - pad, y_mm[1] + pad)

        out = plots_root / f"{_env_tag(env_id)}__glpe_extrinsic_vs_intrinsic.png"
        _save_fig(fig, out)
        plt.close(fig)
        written.append(out)

    return written
