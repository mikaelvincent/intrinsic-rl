from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Sequence

from irl.utils.runs import find_latest_ckpt

_CKPT_STEP_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")

_CKPT_POLICIES = {"latest", "fixed_step", "all_steps", "every_k"}


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


def _normalize_ckpt_policy(policy: str) -> str:
    p = str(policy).strip().lower()
    if p not in _CKPT_POLICIES:
        raise ValueError(
            f"Unknown ckpt_policy={policy!r}. Expected one of: {sorted(_CKPT_POLICIES)}"
        )
    return p


def _list_step_ckpts(run_dir: Path) -> list[tuple[int, Path]]:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return []

    out: list[tuple[int, Path]] = []
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
            out.append((step, p))
    except Exception:
        return []

    out.sort(key=lambda t: (t[0], str(t[1])))
    return out


def _pick_step_at_or_before(step_ckpts: list[tuple[int, Path]], target_step: int) -> Path | None:
    if not step_ckpts:
        return None

    tgt = int(target_step)
    best: Path | None = None
    for s, p in step_ckpts:
        if int(s) <= tgt:
            best = p
        else:
            break

    return best if best is not None else step_ckpts[0][1]


def _downsample_paths(paths: list[Path], max_keep: int | None) -> list[Path]:
    if max_keep is None:
        return paths
    m = int(max_keep)
    if m <= 0 or len(paths) <= m:
        return paths
    if m == 1:
        return [paths[-1]]

    n = int(len(paths))
    if m >= n:
        return paths

    idxs: list[int] = []
    for i in range(m):
        j = int(round((i * (n - 1)) / float(m - 1)))
        idxs.append(j)

    uniq: list[int] = []
    seen: set[int] = set()
    for j in idxs:
        jj = int(max(0, min(n - 1, j)))
        if jj not in seen:
            uniq.append(jj)
            seen.add(jj)

    return [paths[j] for j in uniq]


def select_ckpts_for_run(
    run_dir: Path,
    *,
    policy: str = "latest",
    target_step: int | None = None,
    every_k: int | None = None,
    max_ckpts_per_run: int | None = None,
) -> list[Path]:
    pol = _normalize_ckpt_policy(policy)
    rd = Path(run_dir).resolve()

    if pol == "latest":
        ckpt = find_latest_ckpt(rd)
        return [ckpt] if ckpt is not None else []

    step_ckpts = _list_step_ckpts(rd)
    if not step_ckpts:
        ckpt = find_latest_ckpt(rd)
        return [ckpt] if ckpt is not None else []

    if pol == "fixed_step":
        if target_step is None:
            raise ValueError("ckpt_policy='fixed_step' requires target_step.")
        picked = _pick_step_at_or_before(step_ckpts, int(target_step))
        return [picked] if picked is not None else []

    if pol == "all_steps":
        paths = [p for _s, p in step_ckpts]
        return _downsample_paths(paths, max_ckpts_per_run)

    if pol == "every_k":
        if every_k is None:
            raise ValueError("ckpt_policy='every_k' requires every_k.")
        k = int(every_k)
        if k <= 0:
            raise ValueError("every_k must be >= 1.")

        max_step = int(step_ckpts[-1][0])
        targets = list(range(0, max_step + 1, k))

        chosen: dict[int, Path] = {}
        idx = 0
        best_idx = -1

        for t in targets:
            tt = int(t)
            while idx < len(step_ckpts) and int(step_ckpts[idx][0]) <= tt:
                best_idx = idx
                idx += 1

            if best_idx >= 0:
                s, p = step_ckpts[best_idx]
                chosen[int(s)] = p
            else:
                s0, p0 = step_ckpts[0]
                chosen[int(s0)] = p0

        s_last, p_last = step_ckpts[-1]
        chosen[int(s_last)] = p_last

        paths = [chosen[s] for s in sorted(chosen.keys())]
        return _downsample_paths(paths, max_ckpts_per_run)

    raise ValueError(f"Unsupported ckpt_policy={policy!r}.")


def discover_run_dirs_with_selected_ckpts(
    root: Path,
    *,
    policy: str = "latest",
    target_step: int | None = None,
    every_k: int | None = None,
    max_ckpts_per_run: int | None = None,
) -> list[tuple[Path, Path]]:
    pol = _normalize_ckpt_policy(policy)
    if pol == "latest":
        return discover_run_dirs_with_latest_ckpt(root)

    base = Path(root).resolve()
    if not base.exists():
        return []

    out: list[tuple[Path, Path]] = []
    seen: set[Path] = set()

    for ckpt_dir in base.rglob("checkpoints"):
        if not ckpt_dir.is_dir():
            continue

        run_dir = ckpt_dir.parent.resolve()
        if run_dir in seen:
            continue
        seen.add(run_dir)

        ckpts = select_ckpts_for_run(
            run_dir,
            policy=pol,
            target_step=target_step,
            every_k=every_k,
            max_ckpts_per_run=max_ckpts_per_run,
        )
        for ckpt in ckpts:
            out.append((run_dir, Path(ckpt).resolve()))

    out.sort(key=lambda t: (str(t[0]), str(t[1])))
    return out
