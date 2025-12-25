#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from split_vm_configs import ENV_RUNTIME_MINUTES  # type: ignore
except Exception:
    ENV_RUNTIME_MINUTES = {
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


def _collect_yaml_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.yaml") if p.is_file()])


def _env_key_for_rel_path(rel_path: Path) -> str:
    if len(rel_path.parts) < 2:
        raise ValueError(f"{rel_path.as_posix()}: expected env_dir/file.yaml")
    return rel_path.parts[0]


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


def _parse_seed_list(rel_path: Path, lines: Sequence[str]) -> List[int]:
    seed_line_idx = None
    for i, line in enumerate(lines):
        if _SEED_KEY_RE.match(line.rstrip("\r\n")):
            seed_line_idx = i
            break
    if seed_line_idx is None:
        raise ValueError(f"{rel_path.as_posix()}: missing seed block")

    seeds: List[int] = []
    j = seed_line_idx + 1
    while j < len(lines):
        stripped = lines[j].rstrip("\r\n")
        if stripped.strip() == "":
            break
        m = _SEED_ITEM_RE.match(stripped)
        if not m:
            break
        seeds.append(int(m.group(2)))
        j += 1

    if not seeds:
        raise ValueError(f"{rel_path.as_posix()}: empty seed list")

    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{rel_path.as_posix()}: duplicate seeds in output")

    return seeds


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


def _compute_expected_vm_file_seeds(
    sources: Dict[Path, SourceConfig],
    num_vms: int,
) -> List[Dict[Path, List[int]]]:
    jobs: List[Job] = []
    for rel_path, src in sources.items():
        env_key = _env_key_for_rel_path(rel_path)
        minutes = ENV_RUNTIME_MINUTES.get(env_key)
        if minutes is None:
            raise ValueError(
                f"{rel_path.as_posix()}: unknown env '{env_key}' "
                f"(known: {sorted(ENV_RUNTIME_MINUTES.keys())})"
            )
        for seed in src.seed_order:
            jobs.append(Job(rel_path=rel_path, seed=int(seed), minutes=int(minutes)))

    vm_jobs = _schedule_jobs(jobs, num_vms)

    vm_file_seeds: List[Dict[Path, List[int]]] = []
    for per_vm in vm_jobs:
        per_file: Dict[Path, List[int]] = {}
        for job in per_vm:
            per_file.setdefault(job.rel_path, []).append(job.seed)
        for seeds in per_file.values():
            seeds.sort()
        vm_file_seeds.append(per_file)

    return vm_file_seeds


def _gcd_all(values: Iterable[int]) -> int:
    g = 0
    for v in values:
        g = math.gcd(g, int(v))
    return g


def _diff_snippet(expected: str, actual: str, max_lines: int) -> str:
    exp_lines = expected.splitlines()
    act_lines = actual.splitlines()
    diff = list(
        difflib.unified_diff(
            exp_lines,
            act_lines,
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
    )
    if len(diff) > max_lines:
        diff = diff[:max_lines] + ["... (diff truncated)"]
    return "\n".join(diff)


def _verify_unchanged_except_seed(
    vm_name: str,
    rel_path: Path,
    src: SourceConfig,
    actual_text: str,
    actual_seeds: Sequence[int],
    max_diff_lines: int,
) -> None:
    expected_text = _render_with_seeds(src, actual_seeds)
    if actual_text != expected_text:
        diff = _diff_snippet(expected_text, actual_text, max_diff_lines)
        raise ValueError(
            f"{vm_name}/{rel_path.as_posix()}: content differs outside seed list\n{diff}"
        )


def _verify_global_seed_coverage(
    sources: Dict[Path, SourceConfig],
    used_seed_owner: Dict[Path, Dict[int, str]],
) -> None:
    for rel_path, src in sources.items():
        owners = used_seed_owner.get(rel_path, {})
        got = sorted(owners.keys())
        expected = sorted(src.seed_order)

        if got == expected:
            continue

        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        raise ValueError(
            f"{rel_path.as_posix()}: seed coverage mismatch "
            f"(missing {missing}, extra {extra})"
        )


def _ensure_unique_output_filenames(sources: Dict[Path, SourceConfig]) -> Dict[str, Path]:
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
    return seen


def _vm_output_configs_dir(out_root: Path, vm_name: str) -> Path:
    return out_root / vm_name / "code" / "configs" / vm_name


def _verify_vm_outputs(
    configs_root: Path,
    out_root: Path,
    vm_prefix: str,
    num_vms: int,
    max_diff_lines: int,
    show_loads: bool,
) -> None:
    yaml_paths = _collect_yaml_files(configs_root)
    if not yaml_paths:
        raise ValueError(f"no .yaml files under {configs_root.as_posix()}")

    sources: Dict[Path, SourceConfig] = {}
    for abs_path in yaml_paths:
        src = _load_source_config(configs_root, abs_path)
        sources[src.rel_path] = src

    name_to_src_rel = _ensure_unique_output_filenames(sources)

    expected_by_src = _compute_expected_vm_file_seeds(sources, num_vms)

    used_seed_owner: Dict[Path, Dict[int, str]] = {p: {} for p in sources.keys()}

    vm_loads: List[int] = []
    for i in range(num_vms):
        vm_name = f"{vm_prefix}{i + 1:02d}"
        vm_dir = out_root / vm_name
        cfg_dir = _vm_output_configs_dir(out_root, vm_name)

        if not vm_dir.is_dir():
            raise ValueError(f"missing VM directory: {vm_dir.as_posix()}")
        if not cfg_dir.is_dir():
            raise ValueError(f"missing configs directory: {cfg_dir.as_posix()}")

        expected_src_files = expected_by_src[i]
        expected_files: Dict[Path, List[int]] = {}
        for src_rel, seeds in expected_src_files.items():
            expected_files[Path(src_rel.name)] = seeds

        actual_yaml_paths = _collect_yaml_files(cfg_dir)
        actual_rel_paths = {p.relative_to(cfg_dir) for p in actual_yaml_paths}

        expected_rel_paths = set(expected_files.keys())
        extra = sorted(actual_rel_paths - expected_rel_paths, key=lambda p: p.as_posix())
        missing = sorted(expected_rel_paths - actual_rel_paths, key=lambda p: p.as_posix())
        if extra:
            raise ValueError(
                f"{vm_name}: unexpected output files: "
                + ", ".join(p.as_posix() for p in extra[:5])
                + (" ..." if len(extra) > 5 else "")
            )
        if missing:
            raise ValueError(
                f"{vm_name}: missing output files: "
                + ", ".join(p.as_posix() for p in missing[:5])
                + (" ..." if len(missing) > 5 else "")
            )

        load = 0
        for out_rel_path in sorted(expected_rel_paths, key=lambda p: p.as_posix()):
            out_path = cfg_dir / out_rel_path
            src_rel = name_to_src_rel.get(out_rel_path.name)
            if src_rel is None:
                raise ValueError(f"{vm_name}: unknown config filename {out_rel_path.name}")

            src = sources.get(src_rel)
            if src is None:
                raise ValueError(f"{vm_name}: unknown source config path {src_rel.as_posix()}")

            display_rel_path = Path("code") / "configs" / vm_name / out_rel_path

            actual_text = _read_text_preserve_newlines(out_path)
            actual_lines = actual_text.splitlines(keepends=True)
            actual_seeds = sorted(_parse_seed_list(display_rel_path, actual_lines))
            unknown = sorted(set(actual_seeds) - set(src.seed_order))
            if unknown:
                raise ValueError(
                    f"{vm_name}/{display_rel_path.as_posix()}: unknown seeds {unknown}"
                )

            _verify_unchanged_except_seed(
                vm_name=vm_name,
                rel_path=display_rel_path,
                src=src,
                actual_text=actual_text,
                actual_seeds=actual_seeds,
                max_diff_lines=max_diff_lines,
            )

            expected_seeds = expected_files[out_rel_path]
            if actual_seeds != expected_seeds:
                raise ValueError(
                    f"{vm_name}/{display_rel_path.as_posix()}: seeds mismatch "
                    f"(expected {expected_seeds}, got {actual_seeds})"
                )

            per_file = used_seed_owner.get(src_rel)
            if per_file is None:
                per_file = {}
                used_seed_owner[src_rel] = per_file
            for s in actual_seeds:
                prev = per_file.get(int(s))
                if prev is not None:
                    raise ValueError(
                        f"{vm_name}/{display_rel_path.as_posix()}: seed {s} duplicated "
                        f"(also in {prev})"
                    )
                per_file[int(s)] = vm_name

            env_key = _env_key_for_rel_path(src_rel)
            minutes = ENV_RUNTIME_MINUTES.get(env_key)
            if minutes is None:
                raise ValueError(f"{vm_name}/{display_rel_path.as_posix()}: unknown env '{env_key}'")
            load += len(expected_seeds) * int(minutes)

        vm_loads.append(load)

    _verify_global_seed_coverage(sources, used_seed_owner)

    total_minutes = sum(vm_loads)
    avg = total_minutes / num_vms
    min_load = min(vm_loads)
    max_load = max(vm_loads)
    spread = max_load - min_load

    job_sizes = list(set(int(v) for v in ENV_RUNTIME_MINUTES.values()))
    gcd_minutes = _gcd_all(job_sizes)
    max_job = max(job_sizes) if job_sizes else 0
    ceil_avg_multiple = (
        ((total_minutes + (num_vms * gcd_minutes) - 1) // (num_vms * gcd_minutes))
        * gcd_minutes
        if gcd_minutes > 0
        else math.ceil(avg)
    )
    lower_bound = max(max_job, int(ceil_avg_multiple))

    print(f"Validated VMs: {num_vms}")
    print(f"Validated output files: {sum(len(m) for m in expected_by_src)}")
    print(f"Total workload: {total_minutes / 60:.2f}h")
    print(f"Ideal per-VM: {avg / 60:.2f}h")
    print(f"Min/Max per-VM: {min_load / 60:.2f}h / {max_load / 60:.2f}h")
    print(f"Spread: {spread} minutes")
    print(f"Theoretical LB on max: {lower_bound} minutes")

    if spread == 0 and abs(max_load - avg) < 1e-9:
        print("Workload balance: perfect")
    elif max_load == lower_bound:
        print("Workload balance: max meets lower bound")
    else:
        print("Workload balance: could be improved")

    if show_loads:
        for i, minutes in enumerate(vm_loads):
            vm_name = f"{vm_prefix}{i + 1:02d}"
            print(f"{vm_name}: {minutes / 60:.2f}h")


def main() -> int:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        prog="verify_split_vm_outputs",
        description="Verify outputs created by split_vm_configs.py.",
    )
    parser.add_argument("--num-vms", type=int, default=40)
    parser.add_argument("--configs-dir", type=str, default="configs")
    parser.add_argument("--out-dir", type=str, default="../..")
    parser.add_argument("--vm-prefix", type=str, default="VM_")
    parser.add_argument("--max-diff-lines", type=int, default=120)
    parser.add_argument("--show-loads", action="store_true")
    args = parser.parse_args()

    if args.num_vms <= 0:
        raise ValueError("--num-vms must be > 0")
    if "/" in args.vm_prefix or "\\" in args.vm_prefix:
        raise ValueError("--vm-prefix must be a simple name prefix")
    if args.max_diff_lines <= 0:
        raise ValueError("--max-diff-lines must be > 0")

    configs_root = _resolve_path(base_dir, args.configs_dir)
    out_root = _resolve_path(base_dir, args.out_dir)

    try:
        _verify_vm_outputs(
            configs_root=configs_root,
            out_root=out_root,
            vm_prefix=args.vm_prefix,
            num_vms=args.num_vms,
            max_diff_lines=args.max_diff_lines,
            show_loads=args.show_loads,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
