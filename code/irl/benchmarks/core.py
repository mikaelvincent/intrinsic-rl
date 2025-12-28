from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable

import numpy as np
import torch

from irl.utils.checkpoint import atomic_write_text
from irl.utils.io import atomic_write_csv
from irl.utils.seeding import seed_all as _seed_all


@dataclass
class BenchResult:
    name: str
    metric: str
    unit: str
    params: dict[str, Any]
    values: list[float]
    durations_s: list[float]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.error is not None:
            return {
                "name": str(self.name),
                "metric": str(self.metric),
                "unit": str(self.unit),
                "params": dict(self.params),
                "error": str(self.error),
            }

        vals = [float(v) for v in self.values if math.isfinite(float(v))]
        durs = [float(d) for d in self.durations_s if math.isfinite(float(d))]

        return {
            "name": str(self.name),
            "metric": str(self.metric),
            "unit": str(self.unit),
            "params": dict(self.params),
            "trials": int(len(self.values)),
            "values": [float(v) for v in self.values],
            "durations_s": [float(d) for d in self.durations_s],
            "value_median": float(median(vals)) if vals else float("nan"),
            "value_mean": float(mean(vals)) if vals else float("nan"),
            "value_stdev": float(pstdev(vals)) if len(vals) > 1 else 0.0,
            "duration_median_s": float(median(durs)) if durs else float("nan"),
            "duration_mean_s": float(mean(durs)) if durs else float("nan"),
            "duration_stdev_s": float(pstdev(durs)) if len(durs) > 1 else 0.0,
        }


def _stable_seed(base_seed: int, tag: str) -> int:
    h = hashlib.sha256(f"{int(base_seed)}|{str(tag)}".encode("utf-8")).hexdigest()[:8]
    return int(h, 16) & 0x7FFFFFFF


def _seed_everything(seed: int) -> None:
    _seed_all(seed)


def _maybe_cuda_sync(dev: torch.device) -> None:
    if dev.type != "cuda":
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def _time_loop(fn: Callable[[], None], iters: int, sync: Callable[[], None] | None) -> float:
    if sync is not None:
        sync()
    t0 = time.perf_counter()
    for _ in range(int(max(1, iters))):
        fn()
    if sync is not None:
        sync()
    return float(time.perf_counter() - t0)


def _bench_throughput_with_setup(
    *,
    name: str,
    metric: str,
    unit: str,
    work_per_iter: float,
    iters: int,
    warmup: int,
    trials: int,
    params: dict[str, Any],
    device: torch.device,
    setup_fn: Callable[[int], Callable[[], None]],
) -> BenchResult:
    sync = (lambda: _maybe_cuda_sync(device)) if device.type == "cuda" else None
    durations: list[float] = []
    values: list[float] = []

    for trial_idx in range(int(max(1, trials))):
        fn = setup_fn(int(trial_idx))

        for _ in range(int(max(0, warmup))):
            fn()

        dt = _time_loop(fn, iters=int(iters), sync=sync)
        durations.append(float(dt))
        values.append((float(work_per_iter) * float(iters)) / max(1e-12, float(dt)))

    return BenchResult(
        name=str(name),
        metric=str(metric),
        unit=str(unit),
        params=dict(params),
        values=values,
        durations_s=durations,
        error=None,
    )


def _fmt(x: float) -> str:
    if not math.isfinite(float(x)):
        return "nan"
    ax = abs(float(x))
    if ax >= 1e9:
        return f"{x:.3e}"
    if ax >= 1e6:
        return f"{x:.3e}"
    if ax >= 1e3:
        return f"{x:.3f}"
    return f"{x:.6f}"


def _print_summary(run_meta: dict[str, Any], results: list[BenchResult], outputs: dict[str, str]) -> None:
    dev = str(run_meta.get("device", "cpu"))
    threads = int(run_meta.get("torch_num_threads", 0) or 0)
    seed = int(run_meta.get("seed", 0))
    print(f"[bench] device={dev} threads={threads} seed={seed}")
    print(f"[bench] python={run_meta.get('python')} torch={run_meta.get('torch')} numpy={run_meta.get('numpy')}")
    if run_meta.get("cuda_name"):
        print(f"[bench] cuda={run_meta.get('cuda_name')} (runtime={run_meta.get('cuda_runtime')})")

    print("-" * 96)
    hdr = f"{'benchmark':40}  {'metric':18}  {'median':14}  {'unit':12}  {'trials':5}"
    print(hdr)
    print("-" * 96)

    for r in sorted(results, key=lambda x: x.name):
        if r.error is not None:
            print(f"{r.name:40}  {'error':18}  {'-':14}  {'-':12}  {0:5d}")
            continue

        d = r.to_dict()
        v_med = float(d.get("value_median", float("nan")))
        n = int(d.get("trials", 0))
        print(f"{r.name:40}  {r.metric:18}  {_fmt(v_med):>14}  {r.unit:12}  {n:5d}")

    print("-" * 96)
    if outputs.get("latest_json"):
        print(f"[bench] results_json={outputs['latest_json']}")
    if outputs.get("latest_csv"):
        print(f"[bench] results_csv={outputs['latest_csv']}")


def _system_info(*, device: torch.device, seed: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pid": int(os.getpid()),
        "seed": int(seed),
        "device": str(device),
        "torch": str(getattr(torch, "__version__", "")),
        "numpy": str(getattr(np, "__version__", "")),
        "torch_num_threads": int(torch.get_num_threads()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": str(getattr(torch.version, "cuda", "")),
    }

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        try:
            info["cuda_device_index"] = int(idx)
            info["cuda_name"] = str(torch.cuda.get_device_name(int(idx)))
        except Exception:
            pass

    return info


def run_all_benchmarks(
    *,
    device: str = "cpu",
    threads: int = 1,
    seed: int = 0,
    out_dir: Path = Path("results/benchmarks"),
    quick: bool = False,
) -> dict[str, Any]:
    from . import benchmarks_impl as _impl

    dev = torch.device(str(device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is False.")

    prev_threads = int(torch.get_num_threads())
    torch.set_num_threads(int(max(1, threads)))

    t0 = time.perf_counter()
    try:
        meta = _system_info(device=dev, seed=int(seed))
        meta["quick"] = bool(quick)

        trials = 3 if bool(quick) else 9
        warmup = 1 if bool(quick) else 2
        meta["trials"] = int(trials)
        meta["warmup_iters"] = int(warmup)

        results: list[BenchResult] = []

        try:
            results.append(
                _impl._bench_kdtree_bulk_insert(
                    base_seed=int(seed),
                    device=dev,
                    trials=trials,
                    warmup=warmup,
                    n_points=20000 if bool(quick) else 60000,
                    dim=32,
                    capacity=32,
                    depth_max=10,
                )
            )
        except Exception as exc:
            results.append(
                BenchResult(
                    name="kdtree.bulk_insert",
                    metric="points_per_s",
                    unit="points/s",
                    params={},
                    values=[],
                    durations_s=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

        try:
            results.extend(
                _impl._bench_glpe_pipeline(
                    base_seed=int(seed),
                    device=dev,
                    trials=trials,
                    warmup=warmup,
                    batch_size=8192 if bool(quick) else 16384,
                    iters_compute=3 if bool(quick) else 5,
                    iters_update=2 if bool(quick) else 3,
                    obs_dim=8,
                    n_actions=6,
                    phi_dim=32,
                    hidden=64,
                    region_capacity=32,
                    depth_max=10,
                    gate_cache_interval=64,
                    prefill=0,
                )
            )
        except Exception as exc:
            for suffix in ("compute_batch", "update"):
                results.append(
                    BenchResult(
                        name=f"glpe.{suffix}",
                        metric="transitions_per_s",
                        unit="transitions/s",
                        params={},
                        values=[],
                        durations_s=[],
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        try:
            results.extend(
                _impl._bench_riac_pipeline(
                    base_seed=int(seed),
                    device=dev,
                    trials=trials,
                    warmup=warmup,
                    batch_size=8192 if bool(quick) else 16384,
                    iters_compute=3 if bool(quick) else 5,
                    iters_update=2 if bool(quick) else 3,
                    obs_dim=8,
                    n_actions=6,
                    phi_dim=32,
                    hidden=64,
                    region_capacity=32,
                    depth_max=10,
                    prefill=0,
                )
            )
        except Exception as exc:
            for suffix in ("compute_batch", "update"):
                results.append(
                    BenchResult(
                        name=f"riac.{suffix}",
                        metric="transitions_per_s",
                        unit="transitions/s",
                        params={},
                        values=[],
                        durations_s=[],
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        try:
            results.append(
                _impl._bench_gae(
                    base_seed=int(seed),
                    device=dev,
                    trials=trials,
                    warmup=warmup,
                    iters=30 if bool(quick) else 80,
                    T=128,
                    B=16,
                    obs_dim=8,
                )
            )
        except Exception as exc:
            results.append(
                BenchResult(
                    name="gae.compute_gae",
                    metric="transitions_per_s",
                    unit="transitions/s",
                    params={},
                    values=[],
                    durations_s=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

        try:
            results.append(
                _impl._bench_ppo_update(
                    base_seed=int(seed),
                    device=dev,
                    trials=trials,
                    warmup=warmup,
                    iters=2 if bool(quick) else 5,
                    N=8192 if bool(quick) else 16384,
                    obs_dim=8,
                    n_actions=6,
                    epochs=2,
                    minibatches=8,
                    lr=3e-4,
                )
            )
        except Exception as exc:
            results.append(
                BenchResult(
                    name="ppo.ppo_update",
                    metric="samples_per_s",
                    unit="samples/s",
                    params={},
                    values=[],
                    durations_s=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

        try:
            results.append(
                _impl._bench_env_step(
                    base_seed=int(seed),
                    trials=trials,
                    warmup=0,  # closes env inside timed callable; warmup would reuse a closed env
                    steps=500 if bool(quick) else 1500,
                    env_id="MountainCar-v0",
                    vec_envs=1,
                    frame_skip=1,
                )
            )
            results.append(
                _impl._bench_env_step(
                    base_seed=int(seed),
                    trials=trials,
                    warmup=0,  # closes env inside timed callable; warmup would reuse a closed env
                    steps=250 if bool(quick) else 750,
                    env_id="MountainCar-v0",
                    vec_envs=16,
                    frame_skip=1,
                )
            )
        except Exception as exc:
            for ve in (1, 16):
                results.append(
                    BenchResult(
                        name=f"env.step.sync_vec{ve}",
                        metric="transitions_per_s",
                        unit="transitions/s",
                        params={},
                        values=[],
                        durations_s=[],
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        try:
            results.extend(
                _impl._bench_glpe_gate_median_cache(
                    base_seed=int(seed),
                    device=dev,
                    trials=trials,
                    warmup=warmup,
                    iters=2 if bool(quick) else 4,
                    batch_size=8192 if bool(quick) else 16384,
                    obs_dim=8,
                    n_actions=6,
                    phi_dim=32,
                    hidden=64,
                    region_capacity=32,
                    depth_max=12,
                    prefill=20000 if bool(quick) else 60000,
                    cache_interval=64,
                )
            )
        except Exception as exc:
            for nm, unit in (
                ("glpe.gate_median_cache.baseline", "transitions/s"),
                ("glpe.gate_median_cache.cached", "transitions/s"),
                ("glpe.gate_median_cache.speedup", "x"),
            ):
                results.append(
                    BenchResult(
                        name=nm,
                        metric="error" if nm.endswith("speedup") else "transitions_per_s",
                        unit=unit,
                        params={},
                        values=[],
                        durations_s=[],
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        total_s = float(time.perf_counter() - t0)

        timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        outputs = {
            "latest_json": str((out_dir / "bench_latest.json").resolve()),
            "run_json": str((out_dir / f"bench_{timestamp_tag}.json").resolve()),
            "latest_csv": str((out_dir / "bench_latest.csv").resolve()),
            "run_csv": str((out_dir / f"bench_{timestamp_tag}.csv").resolve()),
        }

        payload = {
            "schema_version": 1,
            "run": meta,
            "total_time_s": float(total_s),
            "results": [r.to_dict() for r in results],
            "outputs": outputs,
        }

        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        atomic_write_text(Path(outputs["run_json"]), text)
        atomic_write_text(Path(outputs["latest_json"]), text)

        cols = [
            "name",
            "metric",
            "unit",
            "trials",
            "value_median",
            "value_mean",
            "value_stdev",
            "duration_median_s",
            "duration_mean_s",
            "duration_stdev_s",
            "params_json",
            "error",
        ]

        csv_rows: list[dict[str, object]] = []
        for r in results:
            d = r.to_dict()
            csv_rows.append(
                {
                    "name": str(d.get("name", "")),
                    "metric": str(d.get("metric", "")),
                    "unit": str(d.get("unit", "")),
                    "trials": int(d.get("trials", 0) or 0) if "trials" in d else 0,
                    "value_median": d.get("value_median", ""),
                    "value_mean": d.get("value_mean", ""),
                    "value_stdev": d.get("value_stdev", ""),
                    "duration_median_s": d.get("duration_median_s", ""),
                    "duration_mean_s": d.get("duration_mean_s", ""),
                    "duration_stdev_s": d.get("duration_stdev_s", ""),
                    "params_json": json.dumps(d.get("params", {}), sort_keys=True),
                    "error": str(d.get("error", "")) if d.get("error") is not None else "",
                }
            )

        atomic_write_csv(Path(outputs["run_csv"]), cols, csv_rows)
        atomic_write_csv(Path(outputs["latest_csv"]), cols, csv_rows)

        _print_summary(meta, results, outputs)
        return payload
    finally:
        torch.set_num_threads(int(max(1, prev_threads)))
