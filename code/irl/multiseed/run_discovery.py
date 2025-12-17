from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from irl.evaluator import evaluate
from irl.pipelines.discovery import collect_ckpts_from_patterns
from irl.utils.checkpoint import load_checkpoint
from irl.utils.runs import find_latest_ckpt as _find_latest_ckpt_impl

from .results import RunResult


def _find_latest_ckpt(run_dir: Path) -> Optional[Path]:
    return _find_latest_ckpt_impl(run_dir)


def _collect_ckpts_from_runs(run_globs: Iterable[str]) -> List[Path]:
    return collect_ckpts_from_patterns(list(run_globs), None)


def _normalize_inputs(runs: Optional[List[str]], ckpts: Optional[List[Path]]) -> List[Path]:
    return collect_ckpts_from_patterns(runs, ckpts)


def _evaluate_ckpt(ckpt: Path, episodes: int, device: str, policy_mode: str = "mode") -> RunResult:
    pm = str(policy_mode).strip().lower()
    if pm not in {"mode", "sample"}:
        raise ValueError("policy_mode must be 'mode' or 'sample'")

    payload = load_checkpoint(ckpt, map_location=device)
    cfg = payload.get("cfg", {}) or {}
    env_id = str(((cfg.get("env") or {}).get("id")) or "MountainCar-v0")
    method = str(cfg.get("method", "vanilla"))
    seed = int(cfg.get("seed", 1))
    step = int(payload.get("step", -1))

    summary = evaluate(env=env_id, ckpt=ckpt, episodes=episodes, device=device, policy_mode=pm)

    return RunResult(
        method=method,
        env_id=summary["env_id"],
        seed=int(summary["seed"]),
        ckpt_path=ckpt,
        ckpt_step=step,
        episodes=int(summary["episodes"]),
        mean_return=float(summary["mean_return"]),
        std_return=float(summary["std_return"]),
        min_return=float(summary["min_return"]),
        max_return=float(summary["max_return"]),
        mean_length=float(summary["mean_length"]),
        std_length=float(summary["std_length"]),
    )
