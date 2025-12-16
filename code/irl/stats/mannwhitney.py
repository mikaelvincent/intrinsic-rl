from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
from typing import Iterable, Literal, Sequence

import numpy as np

Alt = Literal["two-sided", "greater", "less"]


def _as_1d_float(a: Iterable[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(list(a) if not isinstance(a, np.ndarray) else a, dtype=np.float64)
    return x.reshape(-1).astype(np.float64, copy=False)


def rankdata(a: Sequence[float] | np.ndarray) -> np.ndarray:
    x = _as_1d_float(a)
    n = x.size
    if n == 0:
        return np.empty((0,), dtype=np.float64)

    order = np.argsort(x, kind="mergesort")
    xs = x[order]

    ranks_sorted = np.arange(1, n + 1, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * ((i + 1) + j)
            ranks_sorted[i:j] = avg
        i = j

    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


@dataclass(frozen=True)
class MWUResult:
    n_x: int
    n_y: int
    U1: float
    U2: float
    U: float
    z: float
    p_value: float
    cles: float
    cliffs_delta: float
    mean_x: float
    mean_y: float
    median_x: float
    median_y: float


def mannwhitney_u(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    alternative: Alt = "two-sided",
    use_continuity: bool = True,
) -> MWUResult:
    X = _as_1d_float(x)
    Y = _as_1d_float(y)
    n_x = X.size
    n_y = Y.size
    if n_x == 0 or n_y == 0:
        raise ValueError("Both samples must be non-empty.")

    all_vals = np.concatenate([X, Y], axis=0)
    ranks = rankdata(all_vals)
    r_x = ranks[:n_x].sum()

    U1 = r_x - n_x * (n_x + 1) / 2.0
    U2 = n_x * n_y - U1
    U_for_p = min(U1, U2) if alternative == "two-sided" else U1

    N = n_x + n_y
    tie_counts = np.unique(all_vals, return_counts=True)[1]
    tie_term = np.sum(tie_counts**3 - tie_counts)
    var_U = (n_x * n_y / 12.0) * (N + 1.0 - tie_term / (N * (N - 1.0)))

    mean_U = n_x * n_y / 2.0

    if var_U <= 0.0:
        z = 0.0
        p = 1.0
    else:
        cc = 0.0
        if use_continuity:
            cc = 0.5 if (U_for_p > mean_U) else -0.5
            if alternative == "two-sided":
                cc = 0.5

        z = (U_for_p - mean_U + cc) / sqrt(var_U)

        sf = 0.5 * erfc(z / sqrt(2.0))
        cdf = 1.0 - sf

        if alternative == "two-sided":
            p = erfc(abs(z) / sqrt(2.0))
        elif alternative == "greater":
            p = sf
        else:
            p = cdf

    cles = float(U1) / float(n_x * n_y)
    cliffs = 2.0 * cles - 1.0

    return MWUResult(
        n_x=n_x,
        n_y=n_y,
        U1=float(U1),
        U2=float(U2),
        U=float(U_for_p),
        z=float(z),
        p_value=float(p),
        cles=float(cles),
        cliffs_delta=float(cliffs),
        mean_x=float(np.mean(X)),
        mean_y=float(np.mean(Y)),
        median_x=float(np.median(X)),
        median_y=float(np.median(Y)),
    )
