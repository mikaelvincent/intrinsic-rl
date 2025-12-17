from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from irl.evaluator import evaluate
from irl.pipelines.discovery import collect_ckpts_from_patterns
from irl.pipelines.eval import evaluate_ckpt_to_run_result
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

    return evaluate_ckpt_to_run_result(
        ckpt,
        episodes=int(episodes),
        device=str(device),
        policy_mode=pm,
        evaluate_fn=evaluate,
    )
