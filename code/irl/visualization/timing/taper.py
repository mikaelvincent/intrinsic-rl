from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import typer

from irl.visualization.data import aggregate_runs
from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import DPI, FIGSIZE, LEGEND_FRAMEALPHA, LEGEND_FONTSIZE, apply_grid


def _is_effectively_one(vals: np.ndarray, *, tol: float) -> bool:
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return True
    return float(np.max(np.abs(arr - 1.0))) <= float(tol)


def _method_order(methods: list[str]) -> list[str]:
    order = [
        "vanilla",
        "icm",
        "rnd",
        "ride",
        "riac",
        "glpe_lp_only",
        "glpe_impact_only",
        "glpe_nogate",
        "glpe_cache",
        "glpe",
    ]
    idx = {m: i for i, m in enumerate(order)}

    def key(m: str) -> tuple[int, str]:
        ml = str(m).strip().lower()
        if ml in idx:
            return idx[ml], ml
        if ml.startswith("glpe_"):
            return 90, ml
        return 100, ml

    return sorted(list(methods), key=key)


def _interp_to(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    xs = np.asarray(x_src, dtype=np.float64).reshape(-1)
    ys = np.asarray(y_src, dtype=np.float64).reshape(-1)
    xd = np.asarray(x_dst, dtype=np.float64).reshape(-1)
    if xs.size == 0 or ys.size == 0 or xd.size == 0:
        return np.full((int(xd.size),), np.nan, dtype=np.float64)

    n = int(min(xs.size, ys.size))
    xs = xs[:n]
    ys = ys[:n]

    order = np.argsort(xs, kind="mergesort")
    xs = xs[order]
    ys = ys[order]

    return np.interp(xd, xs, ys)


def _read_intrinsic_cfg(run_dir: Path) -> tuple[float | None, float | None]:
    cfg_path = Path(run_dir) / "config.json"
    try:
        text = cfg_path.read_text(encoding="utf-8")
        cfg = json.loads(text)
    except Exception:
        return None, None

    if not isinstance(cfg, dict):
        return None, None

    intrinsic = cfg.get("intrinsic")
    if not isinstance(intrinsic, dict):
        return None, None

    eta = None
    try:
        v = float(intrinsic.get("eta"))
        eta = v if np.isfinite(v) else None
    except Exception:
        eta = None

    r_clip = None
    try:
        v = float(intrinsic.get("r_clip"))
        r_clip = v if np.isfinite(v) else None
    except Exception:
        r_clip = None

    return eta, r_clip


def _median_finite(vals: list[float]) -> float | None:
    if not vals:
        return None
    arr = np.asarray([float(v) for v in vals], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _infer_eta_from_effective(taper: np.ndarray, eta_eff: np.ndarray) -> float | None:
    w = np.asarray(taper, dtype=np.float64).reshape(-1)
    ee = np.asarray(eta_eff, dtype=np.float64).reshape(-1)
    n = int(min(w.size, ee.size))
    if n <= 0:
        return None

    w = w[:n]
    ee = ee[:n]

    m = np.isfinite(w) & np.isfinite(ee) & (w > 1e-6)
    if not bool(np.any(m)):
        return None

    est = ee[m] / w[m]
    est = est[np.isfinite(est)]
    if est.size == 0:
        return None
    return float(np.median(est))


def _grid(n: int) -> tuple[int, int]:
    nn = int(n)
    if nn <= 0:
        return 0, 0
    if nn == 1:
        return 1, 1
    ncols = 1 if nn <= 2 else 2
    nrows = int(math.ceil(float(nn) / float(ncols)))
    return nrows, ncols


def _figsize(nrows: int, ncols: int) -> tuple[float, float]:
    base_w, base_h = float(FIGSIZE[0]), float(FIGSIZE[1])
    w = base_w if int(ncols) <= 1 else base_w * 1.75
    h = base_h * float(max(1, int(nrows)))
    return float(w), float(h)


def plot_intrinsic_taper_weight(
    groups_by_env: Mapping[str, Mapping[str, list[Path]]],
    *,
    plots_root: Path,
    smooth: int = 1,
    shade: bool = True,
    align: str = "interpolate",
    inactive_tol: float = 1e-6,
) -> list[Path]:
    _ = shade
    if not isinstance(groups_by_env, Mapping):
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    align_mode = str(align).strip().lower() or "interpolate"
    if align_mode not in {"union", "intersection", "interpolate"}:
        raise ValueError("align must be one of: union, intersection, interpolate")

    env_recs: list[dict[str, object]] = []

    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: str(kv[0])):
        if not isinstance(by_method, Mapping):
            continue

        glpe_methods: dict[str, list[Path]] = {}
        for m, dirs in by_method.items():
            ml = str(m).strip().lower()
            if ml.startswith("glpe") and isinstance(dirs, (list, tuple)) and dirs:
                glpe_methods[ml] = [Path(p) for p in dirs]

        if not glpe_methods:
            continue

        candidates: list[str] = []
        if "glpe" in glpe_methods:
            candidates.append("glpe")
        for m in _method_order(list(glpe_methods.keys())):
            if m != "glpe":
                candidates.append(m)

        chosen_method: str | None = None
        agg_taper = None

        for m in candidates:
            dirs = glpe_methods.get(m, [])
            if not dirs:
                continue
            try:
                agg = aggregate_runs(
                    dirs,
                    metric="intrinsic_taper_weight",
                    smooth=int(smooth),
                    align=align_mode,
                )
            except Exception:
                continue

            steps = np.asarray(getattr(agg, "steps", np.array([])), dtype=np.int64)
            mean = np.asarray(getattr(agg, "mean", np.array([])), dtype=np.float64)
            if int(getattr(agg, "n_runs", 0) or 0) <= 0 or steps.size == 0 or mean.size == 0:
                continue

            if not _is_effectively_one(mean, tol=float(inactive_tol)):
                chosen_method = str(m)
                agg_taper = agg
                break

        if chosen_method is None or agg_taper is None:
            continue

        run_dirs = glpe_methods[chosen_method]

        try:
            agg_final = aggregate_runs(
                run_dirs,
                metric="r_int_mean",
                smooth=int(smooth),
                align=align_mode,
            )
        except Exception:
            continue

        if int(getattr(agg_final, "n_runs", 0) or 0) <= 0:
            continue

        steps = np.asarray(getattr(agg_taper, "steps", np.array([])), dtype=np.int64)
        taper = np.asarray(getattr(agg_taper, "mean", np.array([])), dtype=np.float64)
        if steps.size == 0 or taper.size == 0:
            continue

        f_steps = np.asarray(getattr(agg_final, "steps", np.array([])), dtype=np.int64)
        f_mean = np.asarray(getattr(agg_final, "mean", np.array([])), dtype=np.float64)
        final = _interp_to(f_steps, f_mean, steps)

        try:
            agg_raw = aggregate_runs(
                run_dirs,
                metric="r_int_raw_mean",
                smooth=int(smooth),
                align=align_mode,
            )
        except Exception:
            agg_raw = None

        raw = None
        if agg_raw is not None and int(getattr(agg_raw, "n_runs", 0) or 0) > 0:
            r_steps = np.asarray(getattr(agg_raw, "steps", np.array([])), dtype=np.int64)
            r_mean = np.asarray(getattr(agg_raw, "mean", np.array([])), dtype=np.float64)
            if r_steps.size and r_mean.size:
                raw = _interp_to(r_steps, r_mean, steps)

        try:
            agg_eta_eff = aggregate_runs(
                run_dirs,
                metric="intrinsic_eta_effective",
                smooth=int(smooth),
                align=align_mode,
            )
        except Exception:
            agg_eta_eff = None

        eta_cfg_vals: list[float] = []
        rclip_cfg_vals: list[float] = []
        for rd in run_dirs:
            eta_v, rc_v = _read_intrinsic_cfg(rd)
            if eta_v is not None:
                eta_cfg_vals.append(float(eta_v))
            if rc_v is not None:
                rclip_cfg_vals.append(float(rc_v))

        eta = _median_finite(eta_cfg_vals)
        r_clip = _median_finite(rclip_cfg_vals)
        r_clip = float(r_clip) if r_clip is not None else 5.0

        if eta is None and agg_eta_eff is not None and int(getattr(agg_eta_eff, "n_runs", 0) or 0) > 0:
            ee_steps = np.asarray(getattr(agg_eta_eff, "steps", np.array([])), dtype=np.int64)
            ee_mean = np.asarray(getattr(agg_eta_eff, "mean", np.array([])), dtype=np.float64)
            eta_eff = _interp_to(ee_steps, ee_mean, steps)
            eta = _infer_eta_from_effective(taper, eta_eff)

        eps = 1e-6
        mask = np.isfinite(taper) & (taper > eps)
        original = np.full_like(final, np.nan, dtype=np.float64)
        original[mask] = final[mask] / taper[mask]

        if raw is not None and eta is not None and np.isfinite(float(eta)):
            fallback = float(eta) * np.clip(raw, -float(r_clip), float(r_clip))
            original[~mask] = fallback[~mask]
        else:
            last = float(original[mask][-1]) if bool(np.any(mask)) else 0.0
            original[~mask] = last

        env_recs.append(
            {
                "env_id": str(env_id),
                "method": str(chosen_method),
                "steps": steps,
                "taper": taper,
                "original": original,
                "final": final,
                "color": _color_for_method(chosen_method),
            }
        )

    if not env_recs:
        return []

    plt = apply_rcparams_paper()
    nrows, ncols = _grid(len(env_recs))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=_figsize(nrows, ncols),
        dpi=int(DPI),
        squeeze=False,
    )
    axes_flat = list(axes.reshape(-1))

    for i, rec in enumerate(env_recs):
        ax = axes_flat[i]
        ax2 = ax.twinx()

        env_id = str(rec["env_id"])
        method = str(rec["method"])
        steps = np.asarray(rec["steps"], dtype=np.int64)
        taper = np.asarray(rec["taper"], dtype=np.float64)
        original = np.asarray(rec["original"], dtype=np.float64)
        final = np.asarray(rec["final"], dtype=np.float64)
        c = str(rec["color"])

        ax.plot(
            steps,
            original,
            linestyle=":",
            linewidth=2.0,
            alpha=0.9,
            color=c,
            zorder=3,
        )
        ax.plot(
            steps,
            final,
            linestyle="-",
            linewidth=2.6,
            alpha=1.0,
            color=c,
            zorder=4,
        )
        ax2.plot(
            steps,
            taper,
            linestyle="--",
            linewidth=1.9,
            alpha=0.85,
            color="black",
            zorder=2,
        )
        ax2.set_ylim(-0.05, 1.05)

        row = int(i // ncols)
        col = int(i % ncols)
        if row == nrows - 1:
            ax.set_xlabel("Environment steps")
        if col == 0:
            ax.set_ylabel("Intrinsic reward (mean)")
        if col == ncols - 1:
            ax2.set_ylabel("Taper ratio")

        ax.set_title(f"{env_id} — {method}")
        apply_grid(ax)

    for j in range(len(env_recs), len(axes_flat)):
        try:
            axes_flat[j].axis("off")
        except Exception:
            pass

    handles = [
        plt.Line2D([], [], color="black", linewidth=2.0, linestyle="--"),
        plt.Line2D([], [], color="black", linewidth=2.0, linestyle=":"),
        plt.Line2D([], [], color="black", linewidth=2.6, linestyle="-"),
    ]
    labels = ["taper ratio", "original intrinsic", "final intrinsic"]

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        framealpha=float(LEGEND_FRAMEALPHA),
        fontsize=int(LEGEND_FONTSIZE),
    )

    fig.suptitle("Intrinsic taper (GLPE family)")
    fig.tight_layout(rect=[0.0, 0.07, 1.0, 0.94])

    out_path = plots_root / "suite__intrinsic_taper.png"
    save_fig_atomic(fig, out_path)
    plt.close(fig)

    typer.echo(f"[suite] Saved taper plot: {out_path}")
    return [out_path]
