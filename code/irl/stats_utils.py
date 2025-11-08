from __future__ import annotations

"""Statistical utilities: Mann–Whitney U and simple bootstrapping.

This module implements:
- rankdata(): average ranks with tie handling (SciPy-like)
- mannwhitney_u(): U statistic + normal-approx p-value with tie correction
- bootstrap_ci(): generic bootstrap percentile CI helper

All implementations rely only on NumPy/stdlib to avoid new dependencies.
"""

from dataclasses import dataclass
from math import erfc, sqrt
from typing import Callable, Iterable, Literal, Sequence, Tuple

import numpy as np


Alt = Literal["two-sided", "greater", "less"]


def _as_1d_float(a: Iterable[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(list(a) if not isinstance(a, np.ndarray) else a, dtype=np.float64)
    return x.reshape(-1).astype(np.float64, copy=False)


def rankdata(a: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return average ranks (1..N) for `a`, using stable sort and tie averaging."""
    x = _as_1d_float(a)
    n = x.size
    if n == 0:
        return np.empty((0,), dtype=np.float64)

    # Stable sort so equal values keep input order before averaging
    order = np.argsort(x, kind="mergesort")
    xs = x[order]

    ranks_sorted = np.arange(1, n + 1, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        # advance j while tied
        while j < n and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * ((i + 1) + j)  # average of ranks i+1 .. j
            ranks_sorted[i:j] = avg
        i = j

    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


@dataclass(frozen=True)
class MWUResult:
    """Result of Mann–Whitney U test and common effect sizes."""

    n_x: int
    n_y: int
    U1: float  # U for X (sum of ranks for X minus n_x*(n_x+1)/2)
    U2: float  # U for Y (n_x*n_y - U1)
    U: float  # U used for p-value (min(U1, U2) for two-sided)
    z: float  # Normal-approx z (with tie & continuity correction)
    p_value: float  # P-value (Alt: two-sided / greater / less)
    cles: float  # Common-language effect size P(X > Y) + 0.5*P(=)
    cliffs_delta: float  # δ = 2*cles - 1  (rank-biserial correlation)
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
    """Mann–Whitney U test (normal approximation with tie correction).

    Parameters
    ----------
    x, y
        Two independent samples (1-D).
    alternative
        "two-sided" (default), "greater" (X tends larger), or "less".
    use_continuity
        Apply ±0.5 continuity correction in z.

    Notes
    -----
    Variance uses the standard tie correction:
        var(U) = n_x n_y / 12 * ( N + 1 - sum(t_i^3 - t_i) / (N (N - 1)) )
    """
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

    if var_U <= 0.0:  # fully tied or degenerate
        z = 0.0
        p = 1.0
    else:
        # Continuity correction based on the direction (for U1 w.r.t mean)
        cc = 0.0
        if use_continuity:
            cc = 0.5 if (U_for_p > mean_U) else (-0.5)
            # For two-sided with min(U1,U2), U_for_p ≤ mean_U; keep sign consistent
            if alternative == "two-sided":
                cc = +0.5  # when using min(U), add 0.5 toward the mean

        z = (U_for_p - mean_U + cc) / sqrt(var_U)

        # Normal tail probabilities (no SciPy): sf(z) = 0.5 * erfc(z / sqrt(2))
        sf = 0.5 * erfc(z / sqrt(2.0))
        cdf = 1.0 - sf

        if alternative == "two-sided":
            p = erfc(abs(z) / sqrt(2.0))
        elif alternative == "greater":
            # H1: X tends larger → large U1 → large z (since U_for_p=U1)
            p = sf
        else:  # "less"
            p = cdf

    cles = float(U1) / float(n_x * n_y)  # includes 0.5 for ties implicitly via ranks
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


def bootstrap_ci(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for a scalar statistic of two samples.

    Returns (stat_point_estimate, lo, hi), where (lo, hi) is a two-sided (100*ci)% interval from bootstrap percentiles.
    """
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
