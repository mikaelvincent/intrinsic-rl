from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic


def _finite_quantiles(values: Any) -> tuple[float, float, float] | None:
    if not isinstance(values, (list, tuple)):
        return None
    try:
        arr = np.asarray([float(v) for v in values], dtype=np.float64).reshape(-1)
    except Exception:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    q25, q50, q75 = np.quantile(arr, [0.25, 0.5, 0.75])
    return float(q25), float(q50), float(q75)


def _style():
    return apply_rcparams_paper()


def _save_fig(fig, path: Path) -> None:
    save_fig_atomic(fig, Path(path))


def _pretty_name(name: str) -> str:
    s = str(name).strip()
    s = s.replace("glpe.gate_median_cache.baseline", "glpe.cache.off")
    s = s.replace("glpe.gate_median_cache.cached", "glpe.cache.on")
    s = s.replace("glpe.gate_median_cache.speedup", "glpe.cache.speedup")
    s = s.replace("glpe.gate_median_cache.", "glpe.cache.")
    return s


def _run_meta_footer(run_meta: Mapping[str, Any] | None) -> str:
    if not isinstance(run_meta, Mapping):
        return ""

    dev = str(run_meta.get("device", "")).strip()
    torch_v = str(run_meta.get("torch", "")).strip()
    py_v = str(run_meta.get("python", "")).strip()

    cuda_name = str(run_meta.get("cuda_name", "")).strip()
    cuda_rt = str(run_meta.get("cuda_runtime", "")).strip()

    threads = run_meta.get("torch_num_threads", None)

    bits: list[str] = []
    if dev:
        bits.append(f"device={dev}")
    if cuda_name:
        cuda_bit = f"cuda={cuda_name}"
        if cuda_rt:
            cuda_bit += f" (cuda {cuda_rt})"
        bits.append(cuda_bit)
    if threads is not None:
        try:
            bits.append(f"threads={int(threads)}")
        except Exception:
            pass
    if torch_v:
        bits.append(f"torch={torch_v}")
    if py_v:
        bits.append(f"python={py_v}")

    return " | ".join(bits)


def _plot_throughput(
    results: list[Mapping[str, Any]], out_path: Path, *, run_meta: Mapping[str, Any] | None = None
) -> bool:
    rows_by_unit: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        if not isinstance(r, Mapping):
            continue
        if r.get("error"):
            continue
        unit = str(r.get("unit", "")).strip()
        if not unit.endswith("/s"):
            continue
        qs = _finite_quantiles(r.get("values"))
        if qs is None:
            continue
        q25, med, q75 = qs
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        rows_by_unit.setdefault(unit, []).append(
            {
                "name": name,
                "label": _pretty_name(name),
                "q25": float(q25),
                "median": float(med),
                "q75": float(q75),
            }
        )

    if not rows_by_unit:
        return False

    plt = _style()
    preferred_units = ["transitions/s", "samples/s", "points/s"]
    units = [u for u in preferred_units if u in rows_by_unit] + sorted(
        [u for u in rows_by_unit.keys() if u not in set(preferred_units)]
    )

    n_units = len(units)
    total_rows = sum(len(rows_by_unit[u]) for u in units)
    height = max(3.0, 1.2 * n_units + 0.28 * total_rows)
    fig, axes = plt.subplots(n_units, 1, figsize=(9.0, height), squeeze=False)

    glpe_color = _color_for_method("glpe")
    other_color = "#1f77b4"

    for ax, unit in zip(axes[:, 0], units):
        rows = list(rows_by_unit[unit])
        rows.sort(key=lambda x: float(x["median"]), reverse=True)

        labels = [r["label"] for r in rows]
        medians = np.asarray([float(r["median"]) for r in rows], dtype=np.float64)
        lo = medians - np.asarray([float(r["q25"]) for r in rows], dtype=np.float64)
        hi = np.asarray([float(r["q75"]) for r in rows], dtype=np.float64) - medians
        lo = np.maximum(lo, 0.0)
        hi = np.maximum(hi, 0.0)

        y = np.arange(len(rows), dtype=np.float64)
        colors = [glpe_color if str(r["name"]).startswith("glpe") else other_color for r in rows]

        ax.barh(y, medians, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.errorbar(
            medians,
            y,
            xerr=np.vstack([lo, hi]),
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=3,
            capthick=0.9,
            alpha=0.9,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        ax.set_title(f"Throughput ({unit})", loc="left", fontweight="bold")
        ax.grid(True, axis="x", alpha=0.25, linestyle="--")

        finite_pos = medians[np.isfinite(medians) & (medians > 0.0)]
        if finite_pos.size >= 2:
            ratio = float(finite_pos.max() / finite_pos.min())
            if ratio >= 50.0:
                ax.set_xscale("log")

    fig.suptitle("Microbenchmarks throughput (median ± IQR)", y=0.995, fontweight="bold")

    footer = _run_meta_footer(run_meta)
    bottom = 0.0
    if footer:
        fig.text(0.01, 0.01, footer, ha="left", va="bottom", fontsize=8, alpha=0.9)
        bottom = 0.04

    fig.tight_layout(rect=[0.0, bottom, 1.0, 0.985])

    _save_fig(fig, Path(out_path))
    plt.close(fig)
    return True


def _get_result_by_name(results: list[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    target = str(name).strip()
    for r in results:
        if not isinstance(r, Mapping):
            continue
        if r.get("error"):
            continue
        if str(r.get("name", "")).strip() == target:
            return r
    return None


def _plot_cache_comparison(
    base: Mapping[str, Any],
    cached: Mapping[str, Any],
    out_path: Path,
    *,
    run_meta: Mapping[str, Any] | None = None,
) -> bool:
    qs_base = _finite_quantiles(base.get("values"))
    qs_cached = _finite_quantiles(cached.get("values"))
    if qs_base is None or qs_cached is None:
        return False

    b_q25, b_med, b_q75 = qs_base
    c_q25, c_med, c_q75 = qs_cached

    if not (np.isfinite(b_med) and np.isfinite(c_med)):
        return False

    unit = str(base.get("unit", "")).strip() or str(cached.get("unit", "")).strip() or "transitions/s"

    base_ci = None
    cached_ci = None
    try:
        p = base.get("params")
        if isinstance(p, Mapping) and p.get("cache_interval") is not None:
            base_ci = int(p.get("cache_interval"))
    except Exception:
        base_ci = None
    try:
        p = cached.get("params")
        if isinstance(p, Mapping) and p.get("cache_interval") is not None:
            cached_ci = int(p.get("cache_interval"))
    except Exception:
        cached_ci = None

    base_label = "no cache" if base_ci is None else f"no cache (interval={base_ci})"
    cached_label = "cached" if cached_ci is None else f"cached (interval={cached_ci})"

    def _method_key(cache_interval: int | None) -> str:
        try:
            return "glpe" if cache_interval is None or int(cache_interval) == 1 else "glpe_cache"
        except Exception:
            return "glpe"

    plt = _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    labels = [base_label, cached_label]
    x = np.arange(2, dtype=np.float64)

    medians = np.asarray([b_med, c_med], dtype=np.float64)
    lo = medians - np.asarray([b_q25, c_q25], dtype=np.float64)
    hi = np.asarray([b_q75, c_q75], dtype=np.float64) - medians
    lo = np.maximum(lo, 0.0)
    hi = np.maximum(hi, 0.0)

    ax.bar(
        x,
        medians,
        color=[_color_for_method(_method_key(base_ci)), _color_for_method(_method_key(cached_ci))],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.errorbar(
        x,
        medians,
        yerr=np.vstack([lo, hi]),
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=3,
        capthick=0.9,
        alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel(unit)

    speed = float("nan")
    if float(b_med) > 0.0 and np.isfinite(float(b_med)) and np.isfinite(float(c_med)):
        speed = float(c_med) / float(b_med)

    title = "Gate median caching comparison (median ± IQR)"
    if np.isfinite(speed):
        title += f" — speedup ≈ {speed:.2f}×"
    ax.set_title(title, loc="left", fontweight="bold")

    ax.grid(True, axis="y", alpha=0.25, linestyle="--")

    finite_pos = medians[np.isfinite(medians) & (medians > 0.0)]
    if finite_pos.size == 2:
        ratio = float(finite_pos.max() / finite_pos.min())
        if ratio >= 50.0:
            ax.set_yscale("log")

    footer = _run_meta_footer(run_meta)
    if footer:
        fig.text(0.01, 0.01, footer, ha="left", va="bottom", fontsize=8, alpha=0.9)
        fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    else:
        fig.tight_layout()

    _save_fig(fig, Path(out_path))
    plt.close(fig)
    return True


def _plot_speedup(
    results: list[Mapping[str, Any]], out_path: Path, *, run_meta: Mapping[str, Any] | None = None
) -> bool:
    base = _get_result_by_name(results, "glpe.gate_median_cache.baseline")
    cached = _get_result_by_name(results, "glpe.gate_median_cache.cached")
    if base is not None and cached is not None:
        if _plot_cache_comparison(base, cached, out_path, run_meta=run_meta):
            return True

    rows: list[dict[str, Any]] = []
    for r in results:
        if not isinstance(r, Mapping):
            continue
        if r.get("error"):
            continue
        unit = str(r.get("unit", "")).strip()
        metric = str(r.get("metric", "")).strip()
        if unit != "x" or metric != "speedup":
            continue

        qs = _finite_quantiles(r.get("values"))
        if qs is None:
            continue
        q25, med, q75 = qs
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        rows.append(
            {
                "label": _pretty_name(name),
                "q25": float(q25),
                "median": float(med),
                "q75": float(q75),
            }
        )

    if not rows:
        return False

    plt = _style()
    rows.sort(key=lambda x: float(x["median"]), reverse=True)

    labels = [r["label"] for r in rows]
    medians = np.asarray([float(r["median"]) for r in rows], dtype=np.float64)
    lo = medians - np.asarray([float(r["q25"]) for r in rows], dtype=np.float64)
    hi = np.asarray([float(r["q75"]) for r in rows], dtype=np.float64) - medians
    lo = np.maximum(lo, 0.0)
    hi = np.maximum(hi, 0.0)

    x = np.arange(len(rows), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.6 + 0.6 * len(rows)), 3.8))

    ax.bar(
        x,
        medians,
        color=_color_for_method("glpe"),
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.errorbar(
        x,
        medians,
        yerr=np.vstack([lo, hi]),
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=3,
        capthick=0.9,
        alpha=0.9,
    )
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.6, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Caching speedup (median ± IQR)", loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")

    footer = _run_meta_footer(run_meta)
    if footer:
        fig.text(0.01, 0.01, footer, ha="left", va="bottom", fontsize=8, alpha=0.9)
        fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    else:
        fig.tight_layout()

    _save_fig(fig, Path(out_path))
    plt.close(fig)
    return True


def write_benchmark_plots(payload: Mapping[str, Any], *, out_dir: Path) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        return {}
    results = payload.get("results")
    if not isinstance(results, list):
        return {}

    run_meta = payload.get("run") if isinstance(payload.get("run"), Mapping) else None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created: dict[str, str] = {}

    throughput_path = out_dir / "bench_latest_throughput.png"
    if _plot_throughput(results, throughput_path, run_meta=run_meta):
        created["throughput"] = str(throughput_path)

    speedup_path = out_dir / "bench_latest_speedup.png"
    if _plot_speedup(results, speedup_path, run_meta=run_meta):
        created["speedup"] = str(speedup_path)

    return created
