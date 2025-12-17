from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Sequence

from irl.utils.runs import find_latest_ckpt

_CKPT_STEP_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")


def discover_run_dirs_with_latest_ckpt(root: Path) -> list[tuple[Path, Path]]:
    root = Path(root).resolve()
    if not root.exists():
        return []

    seen: set[Path] = set()
    out: list[tuple[Path, Path]] = []

    for ckpt_dir in root.rglob("checkpoints"):
        if not ckpt_dir.is_dir():
            continue

        run_dir = ckpt_dir.parent.resolve()
        if run_dir in seen:
            continue

        ckpt = find_latest_ckpt(run_dir)
        if ckpt is None:
            continue

        seen.add(run_dir)
        out.append((run_dir, ckpt))

    out.sort(key=lambda t: str(t[0]))
    return out


def discover_run_dirs_with_step_ckpts(root: Path) -> list[Path]:
    root = Path(root).resolve()
    if not root.exists():
        return []

    seen: set[Path] = set()
    out: list[Path] = []

    for ckpt_dir in root.rglob("checkpoints"):
        if not ckpt_dir.is_dir():
            continue

        try:
            has_any = any(p.is_file() and _CKPT_STEP_RE.match(p.name) for p in ckpt_dir.iterdir())
        except Exception:
            has_any = False

        if not has_any:
            continue

        run_dir = ckpt_dir.parent.resolve()
        if run_dir in seen:
            continue

        seen.add(run_dir)
        out.append(run_dir)

    return sorted(out, key=lambda p: str(p))


def collect_ckpts_from_patterns(
    patterns: Sequence[str] | None,
    explicit_ckpts: Sequence[Path] | None,
) -> list[Path]:
    paths: list[Path] = []

    if patterns:
        for pattern in patterns:
            for hit in glob.glob(str(pattern)):
                p = Path(hit)
                if p.is_file():
                    if p.name.startswith("ckpt_") and p.suffix == ".pt":
                        paths.append(p.resolve())
                    continue

                if p.is_dir():
                    ckpt = find_latest_ckpt(p)
                    if ckpt is not None:
                        paths.append(ckpt.resolve())

    if explicit_ckpts:
        for ckpt in explicit_ckpts:
            p = Path(ckpt)
            if p.exists() and p.is_file():
                paths.append(p.resolve())

    out: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)

    return out
