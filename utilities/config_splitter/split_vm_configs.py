#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ENV_RUNTIME_MINUTES: Dict[str, int] = {
    "ant_v5": 60,
    "bipedalwalker_v3": 30,
    "halfcheetah_v5": 60,
    "humanoid_v5": 180,
    "mountaincar_v0": 30,
}

_SEED_KEY_RE = re.compile(r"^\s*seed:\s*(?:#.*)?$")
_SEED_ITEM_RE = re.compile(r"^(\s*)-\s*(\d+)\s*(?:#.*)?$")


@dataclass(frozen=True)
class SourceConfig:
    rel_path: Path
    abs_path: Path
    lines: Tuple[str, ...]
    seed_items_start: int
    seed_items_end: int
    seed_order: Tuple[int, ...]
    seed_to_line: Dict[int, str]


@dataclass(frozen=True)
class Job:
    rel_path: Path
    seed: int
    minutes: int


def _resolve_path(base: Path, user_path: str) -> Path:
    p = Path(user_path)
    return p if p.is_absolute() else (base / p)


def _read_text_preserve_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as f:
        return f.read()


def _write_text_preserve_newlines(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def _load_source_config(configs_root: Path, abs_path: Path) -> SourceConfig:
    rel_path = abs_path.relative_to(configs_root)
    lines = tuple(_read_text_preserve_newlines(abs_path).splitlines(keepends=True))

    seed_line_idx = None
    for i, line in enumerate(lines):
        if _SEED_KEY_RE.match(line.rstrip("\r\n")):
            seed_line_idx = i
            break
    if seed_line_idx is None:
        raise ValueError(f"{rel_path.as_posix()}: missing seed block")

    seed_to_line: Dict[int, str] = {}
    seed_order: List[int] = []
    j = seed_line_idx + 1
    while j < len(lines):
        stripped = lines[j].rstrip("\r\n")
        if stripped.strip() == "":
            break
        m = _SEED_ITEM_RE.match(stripped)
        if not m:
            break
        seed = int(m.group(2))
        if seed in seed_to_line:
            raise ValueError(f"{rel_path.as_posix()}: duplicate seed {seed}")
        seed_to_line[seed] = lines[j]
        seed_order.append(seed)
        j += 1

    if not seed_order:
        raise ValueError(f"{rel_path.as_posix()}: empty seed list")

    return SourceConfig(
        rel_path=rel_path,
        abs_path=abs_path,
        lines=lines,
        seed_items_start=seed_line_idx + 1,
        seed_items_end=j,
        seed_order=tuple(seed_order),
        seed_to_line=seed_to_line,
    )


def _render_with_seeds(src: SourceConfig, seeds: Iterable[int]) -> str:
    wanted = set(int(s) for s in seeds)
    missing = sorted(wanted - set(src.seed_to_line.keys()))
    if missing:
        raise ValueError(f"{src.rel_path.as_posix()}: unknown seeds {missing}")

    new_items = [src.seed_to_line[s] for s in src.seed_order if s in wanted]
    if not new_items:
        raise ValueError(f"{src.rel_path.as_posix()}: no seeds selected")

    out_lines = list(src.lines[: src.seed_items_start])
    out_lines.extend(new_items)
    out_lines.extend(src.lines[src.seed_items_end :])
    return "".join(out_lines)


def _seed_ranges(seeds: List[int]) -> str:
    if not seeds:
        return ""
    seeds = sorted(set(seeds))
    ranges: List[Tuple[int, int]] = []
    start = prev = seeds[0]
    for s in seeds[1:]:
        if s == prev + 1:
            prev = s
            continue
        ranges.append((start, prev))
        start = prev = s
    ranges.append((start, prev))

    parts: List[str] = []
    for a, b in ranges:
        parts.append(str(a) if a == b else f"{a}-{b}")
    return ", ".join(parts)


def _collect_yaml_files(configs_root: Path) -> List[Path]:
    return sorted([p for p in configs_root.rglob("*.yaml") if p.is_file()])


def _env_key_for_rel_path(rel_path: Path) -> str:
    if len(rel_path.parts) < 2:
        raise ValueError(f"{rel_path.as_posix()}: expected env_dir/file.yaml")
    return rel_path.parts[0]


def _schedule_jobs(jobs: List[Job], num_vms: int) -> List[List[Job]]:
    jobs_sorted = sorted(
        jobs,
        key=lambda j: (-j.minutes, j.rel_path.as_posix(), j.seed),
    )

    vm_loads = [0] * num_vms
    vm_jobs: List[List[Job]] = [[] for _ in range(num_vms)]

    for job in jobs_sorted:
        vm_idx = min(range(num_vms), key=lambda i: vm_loads[i])
        vm_jobs[vm_idx].append(job)
        vm_loads[vm_idx] += job.minutes

    return vm_jobs


def _validate_seed_coverage(
    sources: Dict[Path, SourceConfig],
    vm_file_seeds: List[Dict[Path, List[int]]],
) -> None:
    assigned: Dict[Path, List[int]] = {p: [] for p in sources.keys()}
    for per_vm in vm_file_seeds:
        for rel_path, seeds in per_vm.items():
            assigned[rel_path].extend(seeds)

    for rel_path, src in sources.items():
        got = assigned.get(rel_path, [])
        got_sorted = sorted(got)
        expected_sorted = sorted(src.seed_order)
        if got_sorted != expected_sorted:
            raise ValueError(
                f"{rel_path.as_posix()}: assigned seeds mismatch "
                f"(expected {expected_sorted}, got {got_sorted})"
            )


def _ensure_unique_output_filenames(sources: Dict[Path, SourceConfig]) -> None:
    seen: Dict[str, Path] = {}
    for rel_path in sorted(sources.keys(), key=lambda p: p.as_posix()):
        name = rel_path.name
        prev = seen.get(name)
        if prev is not None and prev != rel_path:
            raise ValueError(
                f"duplicate config filename '{name}' "
                f"(from {prev.as_posix()} and {rel_path.as_posix()})"
            )
        seen[name] = rel_path


def _vm_output_configs_dir(out_root: Path, vm_name: str) -> Path:
    return out_root / vm_name / "code" / "configs" / vm_name


def _vm_output_rel_path(vm_name: str, src_rel_path: Path) -> Path:
    return Path(vm_name) / "code" / "configs" / vm_name / src_rel_path.name


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        prog="split_vm_configs",
        description="Split configs across VMs with balanced wall-clock time.",
    )
    parser.add_argument("--num-vms", type=int, default=40)
    parser.add_argument("--configs-dir", type=str, default="configs")
    parser.add_argument("--out-dir", type=str, default="../..")
    parser.add_argument("--vm-prefix", type=str, default="VM_")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.num_vms <= 0:
        raise ValueError("--num-vms must be > 0")
    if "/" in args.vm_prefix or "\\" in args.vm_prefix:
        raise ValueError("--vm-prefix must be a simple name prefix")

    configs_root = _resolve_path(base_dir, args.configs_dir)
    out_root = _resolve_path(base_dir, args.out_dir)

    yaml_paths = _collect_yaml_files(configs_root)
    if not yaml_paths:
        raise ValueError(f"no .yaml files under {configs_root.as_posix()}")

    sources: Dict[Path, SourceConfig] = {}
    jobs: List[Job] = []
    for abs_path in yaml_paths:
        src = _load_source_config(configs_root, abs_path)
        sources[src.rel_path] = src

        env_key = _env_key_for_rel_path(src.rel_path)
        minutes = ENV_RUNTIME_MINUTES.get(env_key)
        if minutes is None:
            raise ValueError(
                f"{src.rel_path.as_posix()}: unknown env '{env_key}' "
                f"(known: {sorted(ENV_RUNTIME_MINUTES.keys())})"
            )

        for seed in src.seed_order:
            jobs.append(Job(rel_path=src.rel_path, seed=int(seed), minutes=int(minutes)))

    _ensure_unique_output_filenames(sources)

    total_minutes = sum(j.minutes for j in jobs)
    target_minutes = total_minutes / args.num_vms

    vm_jobs = _schedule_jobs(jobs, args.num_vms)

    vm_file_seeds: List[Dict[Path, List[int]]] = []
    vm_loads: List[int] = []
    for per_vm in vm_jobs:
        per_file: Dict[Path, List[int]] = {}
        load = 0
        for job in per_vm:
            per_file.setdefault(job.rel_path, []).append(job.seed)
            load += job.minutes
        for seeds in per_file.values():
            seeds.sort()
        vm_file_seeds.append(per_file)
        vm_loads.append(load)

    _validate_seed_coverage(sources, vm_file_seeds)

    vm_names = [f"{args.vm_prefix}{i + 1:02d}" for i in range(args.num_vms)]
    vm_cfg_dirs = [_vm_output_configs_dir(out_root, name) for name in vm_names]
    if not args.dry_run:
        if args.overwrite:
            # Only remove generated config trees to avoid deleting VM workspaces.
            for cfg_dir in vm_cfg_dirs:
                _remove_tree(cfg_dir)
        else:
            existing = [d.as_posix() for d in vm_cfg_dirs if d.exists()]
            if existing:
                raise ValueError(
                    "output config directories already exist; use --overwrite: "
                    + ", ".join(existing[:5])
                    + (" ..." if len(existing) > 5 else "")
                )

        for cfg_dir in vm_cfg_dirs:
            cfg_dir.mkdir(parents=True, exist_ok=True)

    print(f"Total workload: {total_minutes / 60:.2f}h")
    print(f"Ideal per-VM: {target_minutes / 60:.2f}h")
    print(f"Min/Max per-VM: {min(vm_loads) / 60:.2f}h / {max(vm_loads) / 60:.2f}h")

    for i, per_file in enumerate(vm_file_seeds):
        vm_name = vm_names[i]
        print(f"\n[{vm_name}] {vm_loads[i] / 60:.2f}h")
        for rel_path in sorted(per_file.keys(), key=lambda p: p.as_posix()):
            seeds = per_file[rel_path]
            out_path = _vm_output_rel_path(vm_name, rel_path)
            seeds_str = _seed_ranges(seeds)
            suffix = f" (seeds {seeds_str})" if seeds_str else ""
            display_out_path = Path(args.out_dir) / out_path
            print(f"{display_out_path.as_posix()}{suffix}")

            if args.dry_run:
                continue

            src = sources[rel_path]
            rendered = _render_with_seeds(src, seeds)
            _write_text_preserve_newlines(out_root / out_path, rendered)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
