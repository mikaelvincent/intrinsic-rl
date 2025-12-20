from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def save_trajectory_npz(
    *,
    out_dir: Path,
    env_id: str,
    method: str,
    obs: Sequence[np.ndarray],
    rewards_ext: Sequence[float],
    gates: Sequence[int],
    intrinsic: Sequence[float],
    gate_source: str | None,
    intrinsic_semantics: str | None,
) -> Path | None:
    if not obs:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_tag = str(env_id).replace("/", "-")
    traj_file = out_dir / f"{env_tag}_trajectory.npz"

    method_l = str(method).strip().lower()
    is_glpe_family = method_l.startswith("glpe")

    gate_src_out = gate_source
    if gate_src_out is None:
        gate_src_out = "recomputed" if is_glpe_family else "n/a"

    sem_out = intrinsic_semantics
    if sem_out is None:
        sem_out = "missing_intrinsic" if str(gate_src_out) == "missing_intrinsic" else "none"

    np.savez_compressed(
        traj_file,
        obs=np.stack([np.asarray(o) for o in obs]),
        rewards_ext=np.asarray(list(rewards_ext), dtype=np.float32),
        gates=np.asarray(list(gates), dtype=np.int8),
        intrinsic=np.asarray(list(intrinsic), dtype=np.float32),
        env_id=np.asarray([str(env_id)], dtype=np.str_),
        method=np.asarray([str(method)], dtype=np.str_),
        gate_source=np.asarray([str(gate_src_out)], dtype=np.str_),
        intrinsic_semantics=np.asarray([str(sem_out)], dtype=np.str_),
    )
    return traj_file
