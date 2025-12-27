from __future__ import annotations

import numpy as np


def _as_2d_obs(obs: np.ndarray) -> np.ndarray | None:
    x = np.asarray(obs, dtype=np.float64)
    if x.ndim != 2 or int(x.shape[1]) < 2:
        return None
    return x


def _stable_pca_components(vt: np.ndarray) -> np.ndarray:
    comps = np.asarray(vt[:2], dtype=np.float64, copy=True)

    # Fix component sign so re-runs produce consistent axes.
    for i in range(int(comps.shape[0])):
        j = int(np.argmax(np.abs(comps[i])))
        if float(comps[i, j]) < 0.0:
            comps[i] *= -1.0
    return comps


def _pca_project(
    x: np.ndarray,
    *,
    max_fit_points: int,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    finite = np.isfinite(x).all(axis=1)
    xf = x[finite]
    if int(xf.shape[0]) < 2:
        return None

    n_fit = int(xf.shape[0])
    if n_fit > int(max_fit_points):
        idx = np.linspace(0, n_fit - 1, int(max_fit_points), dtype=np.int64)
        x_fit = xf[idx]
    else:
        x_fit = xf

    mean = x_fit.mean(axis=0)
    std = x_fit.std(axis=0, ddof=0)
    std = np.where(std > 1e-12, std, 1.0)

    x_fit_z = (x_fit - mean) / std

    try:
        _u, s, vt = np.linalg.svd(x_fit_z, full_matrices=False)
    except Exception:
        return None

    if int(vt.shape[0]) < 2:
        return None

    comps = _stable_pca_components(vt)

    denom = max(1, int(x_fit_z.shape[0]) - 1)
    var = (s**2) / float(denom)
    total = float(np.sum(var)) if int(var.size) else 0.0

    v1 = float(var[0] / total) if total > 0.0 and np.isfinite(total) else float("nan")
    v2 = float(var[1] / total) if total > 0.0 and np.isfinite(total) else float("nan")

    x_z = (x - mean) / std
    proj = x_z @ comps.T
    pc1 = np.asarray(proj[:, 0], dtype=np.float64)
    pc2 = np.asarray(proj[:, 1], dtype=np.float64)

    pc1[~finite] = np.nan
    pc2[~finite] = np.nan
    return pc1, pc2, v1, v2


def trajectory_projection(
    env_id: str | None,
    obs: np.ndarray,
    *,
    include_bipedalwalker: bool = True,
    max_pca_points: int = 20000,
) -> tuple[np.ndarray, np.ndarray, str, str, str | None] | None:
    x = _as_2d_obs(obs)
    if x is None:
        return None

    D = int(x.shape[1])
    e = (env_id or "").strip()

    if e.startswith("MountainCar") and D >= 2:
        return x[:, 0], x[:, 1], "position", "velocity", None

    if e.startswith("BipedalWalker") and not bool(include_bipedalwalker):
        return None

    if D == 2:
        return x[:, 0], x[:, 1], "obs[0]", "obs[1]", None

    out = _pca_project(x, max_fit_points=int(max_pca_points))
    if out is None:
        return None
    pc1, pc2, v1, v2 = out

    if np.isfinite(v1) and np.isfinite(v2):
        note = f"PCA(zscore, D={D}, var={100.0 * v1:.1f}%/{100.0 * v2:.1f}%)"
    else:
        note = f"PCA(zscore, D={D})"

    return pc1, pc2, "PC1", "PC2", note
