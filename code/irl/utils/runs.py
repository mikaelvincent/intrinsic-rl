from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Iterable, Optional

_CKPT_STEP_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")
_SEED_TAG_RE = re.compile(r"seed(\d+)")


def _dedup_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        rp = Path(p).resolve()
        if rp not in seen:
            out.append(rp)
            seen.add(rp)
    return out


def parse_run_name(run_dir_name: str | Path) -> dict[str, str]:
    info: dict[str, str] = {}

    name_raw = str(run_dir_name)
    try:
        name = Path(name_raw).name
    except Exception:
        name = name_raw

    parts = name.split("__")
    if len(parts) >= 1:
        info["method"] = parts[0]
    if len(parts) >= 2:
        info["env"] = parts[1]
    if len(parts) >= 3:
        m = _SEED_TAG_RE.match(parts[2])
        if m:
            info["seed"] = m.group(1)
    return info


def find_latest_ckpt(run_dir: Path) -> Optional[Path]:
    rd = Path(run_dir)
    if not rd.exists() or not rd.is_dir():
        return None

    ckpt_dir = rd / "checkpoints"
    latest = ckpt_dir / "ckpt_latest.pt"
    if latest.exists():
        return latest

    if not ckpt_dir.exists():
        return None

    best_step = -1
    best_path: Optional[Path] = None
    try:
        for p in ckpt_dir.iterdir():
            if not p.is_file():
                continue
            m = _CKPT_STEP_RE.match(p.name)
            if not m:
                continue
            try:
                step = int(m.group(1))
            except Exception:
                continue
            if step > best_step:
                best_step = step
                best_path = p
    except Exception:
        return None

    return best_path


def list_step_ckpts(run_dir: Path) -> list[tuple[int, Path]]:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return []

    out: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        m = _CKPT_STEP_RE.match(p.name)
        if not m:
            continue
        try:
            step = int(m.group(1))
        except Exception:
            continue
        out.append((step, p))

    out.sort(key=lambda t: (t[0], str(t[1])))
    return out


def discover_runs_by_logs(root: Path) -> list[Path]:
    base = Path(root).resolve()
    if not base.exists():
        return []

    run_dirs: list[Path] = []
    seen: set[Path] = set()

    for csv_path in base.rglob("logs/scalars.csv"):
        try:
            run_dir = csv_path.parent.parent
        except Exception:
            continue
        if not run_dir.is_dir():
            continue
        rd = run_dir.resolve()
        if rd not in seen:
            seen.add(rd)
            run_dirs.append(rd)

    return sorted(run_dirs, key=lambda p: str(p))


def discover_runs_by_checkpoints(root: Path) -> list[Path]:
    base = Path(root).resolve()
    if not base.exists():
        return []

    seen: set[Path] = set()
    out: list[Path] = []

    for ckpt_dir in base.rglob("checkpoints"):
        if not ckpt_dir.is_dir():
            continue
        try:
            has_any = any(p.is_file() and _CKPT_STEP_RE.match(p.name) for p in ckpt_dir.iterdir())
        except Exception:
            has_any = False

        if not has_any:
            continue

        run_dir = ckpt_dir.parent.resolve()
        if run_dir not in seen:
            seen.add(run_dir)
            out.append(run_dir)

    return sorted(out, key=lambda p: str(p))


def expand_runs_from_patterns(patterns: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for pat in patterns:
        for hit in glob.glob(str(pat)):
            p = Path(hit)
            if p.is_file() and p.name == "scalars.csv" and p.parent.name == "logs":
                dirs.append(p.parent.parent)
            elif p.is_dir():
                if (p / "logs" / "scalars.csv").exists():
                    dirs.append(p)
    return _dedup_paths(dirs)
