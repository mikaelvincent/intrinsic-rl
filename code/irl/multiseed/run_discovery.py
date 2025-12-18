from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from irl.cli.validators import normalize_policy_mode
from irl.evaluator import evaluate
from irl.pipelines.discovery import collect_ckpts_from_patterns
from irl.pipelines.eval import EvalCheckpoint, evaluate_checkpoints
from irl.utils.runs import find_latest_ckpt as _find_latest_ckpt_impl

from .results import RunResult


def _find_latest_ckpt(run_dir: Path) -> Optional[Path]:
    return _find_latest_ckpt_impl(run_dir)


def _collect_ckpts_from_runs(run_globs: Iterable[str]) -> List[Path]:
    return collect_ckpts_from_patterns(list(run_globs), None)


def _normalize_inputs(runs: Optional[List[str]], ckpts: Optional[List[Path]]) -> List[Path]:
    return collect_ckpts_from_patterns(runs, ckpts)


def _evaluate_ckpt(ckpt: Path, episodes: int, device: str, policy_mode: str = "mode") -> RunResult:
    pm = normalize_policy_mode(policy_mode, allowed=("mode", "sample"), name="policy_mode")
    results = evaluate_checkpoints(
        [EvalCheckpoint(ckpt=Path(ckpt))],
        episodes=int(episodes),
        device=str(device),
        policy_mode=str(pm),
        evaluate_fn=evaluate,
        skip_failures=False,
    )
    if not results:
        raise RuntimeError(f"Evaluation returned no result for checkpoint: {ckpt}")
    return results[0]
