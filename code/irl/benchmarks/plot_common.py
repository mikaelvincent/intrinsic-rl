from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic


def finite_quantiles(values: Any) -> tuple[float, float, float] | None:
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


def style():
    return apply_rcparams_paper()


def save_fig(fig, path: Path) -> None:
    save_fig_atomic(fig, Path(path))


def pretty_name(name: str) -> str:
    s = str(name).strip()
    s = s.replace("glpe.gate_median_cache.baseline", "glpe.cache.off")
    s = s.replace("glpe.gate_median_cache.cached", "glpe.cache.on")
    s = s.replace("glpe.gate_median_cache.speedup", "glpe.cache.speedup")
    s = s.replace("glpe.gate_median_cache.", "glpe.cache.")
    return s


def run_meta_footer(run_meta: Mapping[str, Any] | None) -> str:
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


def get_result_by_name(results: list[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    target = str(name).strip()
    for r in results:
        if not isinstance(r, Mapping):
            continue
        if r.get("error"):
            continue
        if str(r.get("name", "")).strip() == target:
            return r
    return None
