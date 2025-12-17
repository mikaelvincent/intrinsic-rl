from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from irl.results.summary import RunResult
from irl.utils.checkpoint import load_checkpoint


def _cfg_fields(payload: Mapping[str, Any]) -> tuple[str | None, str | None, int | None]:
    cfg = payload.get("cfg") or {}
    if not isinstance(cfg, Mapping):
        return None, None, None

    env_id = None
    env_cfg = cfg.get("env") or {}
    if isinstance(env_cfg, Mapping) and env_cfg.get("id") is not None:
        env_id = str(env_cfg.get("id"))

    method = str(cfg.get("method")) if cfg.get("method") is not None else None

    seed = None
    if cfg.get("seed") is not None:
        try:
            seed = int(cfg.get("seed"))
        except Exception:
            seed = None

    return env_id, method, seed


def _episode_seeds_hash(summary: Mapping[str, Any]) -> str:
    seeds = summary.get("episode_seeds")
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
        return ""
    try:
        ints = [int(s) for s in seeds]
    except Exception:
        return ""
    if not ints:
        return ""
    blob = ",".join(str(s) for s in ints).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def evaluate_ckpt_to_run_result(
    ckpt: Path,
    *,
    episodes: int,
    device: str,
    policy_mode: str = "mode",
    env: str | None = None,
    method: str | None = None,
    seed: int | None = None,
    save_traj: bool = False,
    traj_out_dir: Path | None = None,
    seed_offset: int = 0,
    episode_seeds: Sequence[int] | None = None,
    payload: Mapping[str, Any] | None = None,
    evaluate_fn: Callable[..., dict] | None = None,
) -> RunResult:
    ckpt_path = Path(ckpt)

    if payload is None:
        payload = load_checkpoint(ckpt_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a mapping.")

    cfg_env_id, cfg_method, cfg_seed = _cfg_fields(payload)

    env_eff = str(env if env is not None else (cfg_env_id or "MountainCar-v0"))
    method_eff = str(method if method is not None else (cfg_method or "vanilla"))

    seed_eff: int | None
    if seed is not None:
        seed_eff = int(seed)
    elif cfg_seed is not None:
        seed_eff = int(cfg_seed)
    else:
        seed_eff = None

    step_payload = -1
    try:
        if payload.get("step") is not None:
            step_payload = int(payload.get("step"))
    except Exception:
        step_payload = -1

    if evaluate_fn is None:
        from irl.evaluator import evaluate as _evaluate

        evaluate_fn = _evaluate

    pm = str(policy_mode).strip().lower() or "mode"

    summary = evaluate_fn(
        env=env_eff,
        ckpt=ckpt_path,
        episodes=int(episodes),
        device=str(device),
        save_traj=bool(save_traj),
        traj_out_dir=traj_out_dir,
        policy_mode=pm,
        episode_seeds=None if episode_seeds is None else list(episode_seeds),
        seed_offset=int(seed_offset),
    )
    if not isinstance(summary, Mapping):
        raise TypeError("evaluate() must return a mapping.")
    if "mean_return" not in summary:
        raise ValueError(
            f"Unsupported evaluation summary format for policy_mode={str(policy_mode)!r}."
        )

    ckpt_step = step_payload
    try:
        if summary.get("checkpoint_step") is not None:
            ckpt_step = int(summary.get("checkpoint_step"))
    except Exception:
        ckpt_step = step_payload

    summary_seed = None
    try:
        if summary.get("seed") is not None:
            summary_seed = int(summary.get("seed"))
    except Exception:
        summary_seed = None

    seed_final = (
        int(seed_eff)
        if seed_eff is not None
        else int(summary_seed) if summary_seed is not None else 0
    )

    return RunResult(
        method=str(method_eff),
        env_id=str(summary.get("env_id", env_eff)),
        seed=int(seed_final),
        ckpt_path=ckpt_path,
        ckpt_step=int(ckpt_step),
        episodes=int(summary.get("episodes", int(episodes))),
        mean_return=float(summary.get("mean_return", 0.0)),
        std_return=float(summary.get("std_return", 0.0)),
        min_return=float(summary.get("min_return", 0.0)),
        max_return=float(summary.get("max_return", 0.0)),
        mean_length=float(summary.get("mean_length", 0.0)),
        std_length=float(summary.get("std_length", 0.0)),
        policy_mode=str(pm),
        seed_offset=int(seed_offset),
        episode_seeds_hash=_episode_seeds_hash(summary),
    )
