from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

from .mannwhitney import _as_1d_float


def bootstrap_ci(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    X = _as_1d_float(x)
    Y = _as_1d_float(y)
    if X.size == 0 or Y.size == 0:
        raise ValueError("Both samples must be non-empty.")

    rng = rng or np.random.default_rng()
    point = float(stat_fn(X, Y))
    if n_boot <= 0:
        return point, float("nan"), float("nan")

    bx = rng.integers(0, X.size, size=(n_boot, X.size))
    by = rng.integers(0, Y.size, size=(n_boot, Y.size))
    stats = np.empty((n_boot,), dtype=np.float64)
    for i in range(n_boot):
        stats[i] = stat_fn(X[bx[i]], Y[by[i]])

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1.0 - alpha))
    return point, lo, hi
