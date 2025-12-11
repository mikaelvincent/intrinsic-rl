"""Checkpoint discovery and normalization helpers for multi-seed evaluation."""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Iterable, List, Optional

from irl.evaluator import evaluate
from irl.utils.checkpoint import load_checkpoint

from .results import RunResult

# Local regex to identify step-numbered checkpoints
_CKPT_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")


def _find_latest_ckpt(run_dir: Path) -> Optional[Path]:
    """Return the latest checkpoint path inside a run directory, if any."""
    if not run_dir.exists() or not run_dir.is_dir():
        return None
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    latest = ckpt_dir / "ckpt_latest.pt"
    if latest.exists():
        return latest

    # Fall back to highest step filename
    candidates: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        m = _CKPT_RE.match(p.name)
        if not m:
            continue
        try:
            step = int(m.group(1))
            candidates.append((step, p))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _collect_ckpts_from_runs(run_globs: Iterable[str]) -> List[Path]:
    """Expand globs and return latest checkpoints from each matching run directory.

    Each entry in ``run_globs`` may be:

    * A glob pattern pointing at run directories.
    * A concrete run directory.
    * A direct checkpoint path (``ckpt_step_*.pt``).

    All are normalised into checkpoint paths, with duplicates removed.
    """
    out: list[Path] = []
    for pattern in run_globs:
        for p in glob.glob(pattern):
            rd = Path(p)
            if rd.is_file():
                # Allow passing a checkpoint file via --runs; accept it directly.
                if rd.name.startswith("ckpt_") and rd.suffix == ".pt":
                    out.append(rd.resolve())
                continue
            ck = _find_latest_ckpt(rd)
            if ck is not None:
                out.append(ck.resolve())
    # De-duplicate while preserving order
    seen = set()
    unique: list[Path] = []
    for p in out:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _normalize_inputs(runs: Optional[List[str]], ckpts: Optional[List[Path]]) -> List[Path]:
    """Combine checkpoints gathered from run globs and explicit --ckpt paths.

    ``runs`` is a list of patterns and/or concrete run directories. On shells
    that expand wildcards eagerly (notably Windows ``cmd.exe`` in some
    configurations), a single pattern argument may arrive as multiple concrete
    directories; this helper treats both cases uniformly.
    """
    paths: list[Path] = []
    if runs:
        paths.extend(_collect_ckpts_from_runs(runs))
    if ckpts:
        for c in ckpts:
            if c.exists() and c.is_file():
                paths.append(c.resolve())
    # De-duplicate
    uniq: list[Path] = []
    seen = set()
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _evaluate_ckpt(ckpt: Path, episodes: int, device: str) -> RunResult:
    """Load metadata from the checkpoint, resolve env/method/seed, and evaluate."""
    payload = load_checkpoint(ckpt, map_location=device)
    cfg = payload.get("cfg", {}) or {}
    env_id = str(((cfg.get("env") or {}).get("id")) or "MountainCar-v0")
    method = str(cfg.get("method", "vanilla"))
    seed = int(cfg.get("seed", 1))
    step = int(payload.get("step", -1))

    summary = evaluate(env=env_id, ckpt=ckpt, episodes=episodes, device=device)

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
