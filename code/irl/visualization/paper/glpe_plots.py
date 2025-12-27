from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..palette import color_for_method as _color_for_method
from ..plot_utils import apply_rcparams_paper, save_fig_atomic
from ..trajectory_projection import trajectory_projection


def _style():
    return apply_rcparams_paper()


def _save_fig(fig, path: Path) -> None:
    save_fig_atomic(fig, Path(path))


def _env_tag(env_id: str) -> str:
    return str(env_id).replace("/", "-")


def _finite_minmax(vals: list[float]) -> tuple[float, float] | None:
    arr = np.asarray([float(v) for v in vals], dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.min()), float(arr.max())


def _npz_str(data: Any, key: str) -> str | None:
    try:
        arr = np.asarray(data[key])
        if arr.size == 0:
            return None
        return str(arr.reshape(-1)[0])
    except Exception:
        return None


def _stable_u32(*parts: str) -> int:
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:8], 16)


def _sample_seed(tag: str, env_id: str) -> int:
    return int(_stable_u32(str(tag).strip(), str(env_id).strip()))


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
        proj = trajectory_projection(env_id, obs_cat, include_bipedalwalker=True)
        if proj is None:
            continue

        x, y, xlab, ylab, proj_note = proj
        g = gates_cat >= 0.5

        finite = np.isfinite(x) & np.isfinite(y)
        if not bool(finite.any()):
            continue

        x = np.asarray(x[finite], dtype=np.float64)
        y = np.asarray(y[finite], dtype=np.float64)
        g = np.asarray(g[finite], dtype=bool)

        idx = _sample_idx(x.shape[0], int(max_points), seed=_sample_seed("glpe_gate_map", env_id))
        x = x[idx]
        y = y[idx]
        g = g[idx]

        fig, ax = plt.subplots(figsize=(7.8, 6.0))

        glpe_c = _color_for_method("glpe")

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
                c=glpe_c,
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

        if proj_note:
            fig.text(0.01, 0.01, str(proj_note), ha="left", va="bottom", fontsize=8, alpha=0.9)

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

        idx = _sample_idx(x.shape[0], int(max_points), seed=_sample_seed("glpe_extint", env_id))
        x = x[idx]
        y = y[idx]
        g = g[idx]

        x_mm = _finite_minmax(x.tolist())
        y_mm = _finite_minmax(y.tolist())
        if x_mm is None or y_mm is None:
            continue

        fig, ax = plt.subplots(figsize=(6.8, 5.6))

        glpe_c = _color_for_method("glpe")

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
                c=glpe_c,
                edgecolor="none",
                label="Active/On",
            )

        ax.axhline(0.0, linewidth=1.0, alpha=0.4, color="black")
        ax.axvline(0.0, linewidth=1.0, alpha=0.4, color="black")

        ax.set_xlabel("Extrinsic reward (per step)")
        ax.set_ylabel("Intrinsic reward (per step)")
        ax.set_title(
            f"{env_id} — GLPE extrinsic vs intrinsic (eval)", loc="left", fontweight="bold"
        )
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
