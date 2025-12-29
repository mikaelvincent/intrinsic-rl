from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

_CKPT_STEP_DIR_RE = re.compile(r"^ckpt_step_(\d+)$")


def _ckpt_step_from_dir(dir_name: str) -> int:
    s = str(dir_name).strip()
    m = _CKPT_STEP_DIR_RE.match(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return -1
    if s == "ckpt_latest":
        return -1
    return -1


def _npz_str(data: object, key: str) -> str | None:
    try:
        arr = np.asarray(getattr(data, "__getitem__")(key))  # type: ignore[misc]
        if arr.size == 0:
            return None
        return str(arr.reshape(-1)[0])
    except Exception:
        return None


def _traj_run_and_ckpt_tag(traj_root: Path, p: Path) -> tuple[str | None, str | None]:
    try:
        rel = p.resolve().relative_to(Path(traj_root).resolve())
    except Exception:
        return None, None
    parts = rel.parts
    if len(parts) < 2:
        return None, None
    return str(parts[0]), str(parts[1])


def select_latest_glpe_trajectories(traj_root: Path) -> list[tuple[str, str, int, Path]]:
    root = Path(traj_root)
    if not root.exists():
        return []

    candidates: list[tuple[str, str, int, Path]] = []
    for p in sorted(root.rglob("*_trajectory.npz"), key=lambda x: str(x)):
        run_name, ckpt_tag = _traj_run_and_ckpt_tag(root, p)
        if run_name is None or ckpt_tag is None:
            continue

        ckpt_step = _ckpt_step_from_dir(str(ckpt_tag))

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

        candidates.append((str(env_id), str(run_name), int(ckpt_step), Path(p)))

    best_by_run: dict[str, tuple[str, str, int, Path]] = {}
    for env_id, run_name, ckpt_step, path in candidates:
        prev = best_by_run.get(run_name)
        if prev is None:
            best_by_run[run_name] = (env_id, run_name, int(ckpt_step), path)
            continue

        prev_step = int(prev[2])
        if int(ckpt_step) > prev_step:
            best_by_run[run_name] = (env_id, run_name, int(ckpt_step), path)
            continue
        if int(ckpt_step) == prev_step and str(path) > str(prev[3]):
            best_by_run[run_name] = (env_id, run_name, int(ckpt_step), path)

    return [best_by_run[k] for k in sorted(best_by_run.keys())]


def _stable_u32(*parts: str) -> int:
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:8], 16)


def sample_seed(tag: str, env_id: str) -> int:
    return int(_stable_u32(str(tag).strip(), str(env_id).strip()))


def sample_idx(n: int, k: int, *, seed: int) -> np.ndarray:
    nn = int(n)
    kk = int(k)
    if kk <= 0 or nn <= kk:
        return np.arange(nn, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return rng.choice(nn, size=kk, replace=False).astype(np.int64, copy=False)
