from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from irl.utils.checkpoint import atomic_replace


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


def _pretty_name(name: str) -> str:
    s = str(name).strip()
    s = s.replace("glpe.gate_median_cache.", "glpe.cache.")
    return s


def _plot_throughput(results: list[Mapping[str, Any]], out_path: Path) -> bool:
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
        colors = ["#d62728" if str(r["name"]).startswith("glpe") else "#1f77b4" for r in rows]

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
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])

    _save_fig(fig, Path(out_path))
    plt.close(fig)
    return True


def _plot_speedup(results: list[Mapping[str, Any]], out_path: Path) -> bool:
    rows: list[dict[str, Any]] = []
    for r in results:
        if not isinstance(r, Mapping):
            continue
        if r.get("error"):
            continue
        unit = str(r.get("unit", "")).strip()
        metric = str(r.get("metric", "")).strip()
        if unit != "x" and metric != "speedup":
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

    ax.bar(x, medians, color="#d62728", alpha=0.85, edgecolor="white", linewidth=0.5)
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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created: dict[str, str] = {}

    throughput_path = out_dir / "bench_latest_throughput.png"
    if _plot_throughput(results, throughput_path):
        created["throughput"] = str(throughput_path)

    speedup_path = out_dir / "bench_latest_speedup.png"
    if _plot_speedup(results, speedup_path):
        created["speedup"] = str(speedup_path)

    return created
