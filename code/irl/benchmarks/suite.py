from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
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
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(s)
        except Exception:
            pass


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


def _bench_kdtree_bulk_insert(
    *,
    base_seed: int,
    device: torch.device,
    trials: int,
    warmup: int,
    n_points: int,
    dim: int,
    capacity: int,
    depth_max: int,
) -> BenchResult:
    from irl.intrinsic.regions import KDTreeRegionStore

    name = "kdtree.bulk_insert"
    seed = _stable_seed(base_seed, name)
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((int(n_points), int(dim))).astype(np.float32)

    def setup_fn(_trial_idx: int) -> Callable[[], None]:
        def run_once() -> None:
            store = KDTreeRegionStore(dim=int(dim), capacity=int(capacity), depth_max=int(depth_max))
            _ = store.bulk_insert(pts)

        return run_once

    return _bench_throughput_with_setup(
        name=name,
        metric="points_per_s",
        unit="points/s",
        work_per_iter=float(n_points),
        iters=1,
        warmup=warmup,
        trials=trials,
        params={
            "n_points": int(n_points),
            "dim": int(dim),
            "capacity": int(capacity),
            "depth_max": int(depth_max),
        },
        device=device,
        setup_fn=setup_fn,
    )


def _make_vec_batch(rng: np.random.Generator, *, n: int, obs_dim: int, n_actions: int):
    obs = rng.standard_normal((int(n), int(obs_dim))).astype(np.float32)
    next_obs = rng.standard_normal((int(n), int(obs_dim))).astype(np.float32)
    actions = rng.integers(0, int(n_actions), size=(int(n),), endpoint=False, dtype=np.int64)
    return obs, next_obs, actions


def _prefill_intrinsic(mod: object, rng: np.random.Generator, *, transitions: int, batch: int, obs_dim: int, n_actions: int):
    remaining = int(max(0, transitions))
    if remaining <= 0:
        return
    b = int(max(1, batch))
    while remaining > 0:
        n = int(min(b, remaining))
        obs, next_obs, actions = _make_vec_batch(rng, n=n, obs_dim=obs_dim, n_actions=n_actions)
        _ = getattr(mod, "compute_batch")(obs, next_obs, actions, reduction="none")
        remaining -= n


def _bench_glpe_pipeline(
    *,
    base_seed: int,
    device: torch.device,
    trials: int,
    warmup: int,
    batch_size: int,
    iters_compute: int,
    iters_update: int,
    obs_dim: int,
    n_actions: int,
    phi_dim: int,
    hidden: int,
    region_capacity: int,
    depth_max: int,
    gate_cache_interval: int,
    prefill: int,
) -> list[BenchResult]:
    import gymnasium as gym

    from irl.intrinsic.glpe import GLPE
    from irl.intrinsic.icm import ICMConfig

    dev_s = str(device)
    name_base = "glpe"
    seed = _stable_seed(base_seed, "glpe.pipeline")
    _seed_everything(seed)
    rng = np.random.default_rng(seed)

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(int(obs_dim),), dtype=np.float32)
    act_space = gym.spaces.Discrete(int(n_actions))
    icm_cfg = ICMConfig(phi_dim=int(phi_dim), hidden=(int(hidden), int(hidden)))

    obs, next_obs, actions = _make_vec_batch(rng, n=int(batch_size), obs_dim=int(obs_dim), n_actions=int(n_actions))

    def _setup_common(_trial_idx: int) -> GLPE:
        _seed_everything(seed + int(_trial_idx))
        mod = GLPE(
            obs_space,
            act_space,
            device=dev_s,
            icm_cfg=icm_cfg,
            region_capacity=int(region_capacity),
            depth_max=int(depth_max),
            normalize_inside=True,
            gating_enabled=True,
        )
        if hasattr(mod, "gate_median_cache_interval"):
            mod.gate_median_cache_interval = int(max(1, gate_cache_interval))
        _prefill_intrinsic(
            mod,
            rng,
            transitions=int(prefill),
            batch=int(batch_size),
            obs_dim=int(obs_dim),
            n_actions=int(n_actions),
        )
        return mod

    def setup_compute(trial_idx: int) -> Callable[[], None]:
        mod = _setup_common(trial_idx)

        def run_once() -> None:
            _ = mod.compute_batch(obs, next_obs, actions, reduction="none")

        return run_once

    def setup_update(trial_idx: int) -> Callable[[], None]:
        mod = _setup_common(trial_idx)

        def run_once() -> None:
            _ = mod.update(obs, next_obs, actions, steps=1)

        return run_once

    compute_res = _bench_throughput_with_setup(
        name=f"{name_base}.compute_batch",
        metric="transitions_per_s",
        unit="transitions/s",
        work_per_iter=float(batch_size),
        iters=int(iters_compute),
        warmup=warmup,
        trials=trials,
        params={
            "device": dev_s,
            "batch": int(batch_size),
            "obs_dim": int(obs_dim),
            "n_actions": int(n_actions),
            "phi_dim": int(phi_dim),
            "hidden": int(hidden),
            "region_capacity": int(region_capacity),
            "depth_max": int(depth_max),
            "gate_cache_interval": int(gate_cache_interval),
            "prefill": int(prefill),
        },
        device=device,
        setup_fn=setup_compute,
    )

    update_res = _bench_throughput_with_setup(
        name=f"{name_base}.update",
        metric="transitions_per_s",
        unit="transitions/s",
        work_per_iter=float(batch_size),
        iters=int(iters_update),
        warmup=0,
        trials=trials,
        params={
            "device": dev_s,
            "batch": int(batch_size),
            "obs_dim": int(obs_dim),
            "n_actions": int(n_actions),
            "phi_dim": int(phi_dim),
            "hidden": int(hidden),
            "region_capacity": int(region_capacity),
            "depth_max": int(depth_max),
            "gate_cache_interval": int(gate_cache_interval),
            "prefill": int(prefill),
        },
        device=device,
        setup_fn=setup_update,
    )

    return [compute_res, update_res]


def _bench_riac_pipeline(
    *,
    base_seed: int,
    device: torch.device,
    trials: int,
    warmup: int,
    batch_size: int,
    iters_compute: int,
    iters_update: int,
    obs_dim: int,
    n_actions: int,
    phi_dim: int,
    hidden: int,
    region_capacity: int,
    depth_max: int,
    prefill: int,
) -> list[BenchResult]:
    import gymnasium as gym

    from irl.intrinsic.icm import ICMConfig
    from irl.intrinsic.riac import RIAC

    dev_s = str(device)
    name_base = "riac"
    seed = _stable_seed(base_seed, "riac.pipeline")
    _seed_everything(seed)
    rng = np.random.default_rng(seed)

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(int(obs_dim),), dtype=np.float32)
    act_space = gym.spaces.Discrete(int(n_actions))
    icm_cfg = ICMConfig(phi_dim=int(phi_dim), hidden=(int(hidden), int(hidden)))

    obs, next_obs, actions = _make_vec_batch(rng, n=int(batch_size), obs_dim=int(obs_dim), n_actions=int(n_actions))

    def _setup_common(_trial_idx: int) -> RIAC:
        _seed_everything(seed + int(_trial_idx))
        mod = RIAC(
            obs_space,
            act_space,
            device=dev_s,
            icm_cfg=icm_cfg,
            region_capacity=int(region_capacity),
            depth_max=int(depth_max),
        )
        _prefill_intrinsic(
            mod,
            rng,
            transitions=int(prefill),
            batch=int(batch_size),
            obs_dim=int(obs_dim),
            n_actions=int(n_actions),
        )
        return mod

    def setup_compute(trial_idx: int) -> Callable[[], None]:
        mod = _setup_common(trial_idx)

        def run_once() -> None:
            _ = mod.compute_batch(obs, next_obs, actions, reduction="none")

        return run_once

    def setup_update(trial_idx: int) -> Callable[[], None]:
        mod = _setup_common(trial_idx)

        def run_once() -> None:
            _ = mod.update(obs, next_obs, actions, steps=1)

        return run_once

    compute_res = _bench_throughput_with_setup(
        name=f"{name_base}.compute_batch",
        metric="transitions_per_s",
        unit="transitions/s",
        work_per_iter=float(batch_size),
        iters=int(iters_compute),
        warmup=warmup,
        trials=trials,
        params={
            "device": dev_s,
            "batch": int(batch_size),
            "obs_dim": int(obs_dim),
            "n_actions": int(n_actions),
            "phi_dim": int(phi_dim),
            "hidden": int(hidden),
            "region_capacity": int(region_capacity),
            "depth_max": int(depth_max),
            "prefill": int(prefill),
        },
        device=device,
        setup_fn=setup_compute,
    )

    update_res = _bench_throughput_with_setup(
        name=f"{name_base}.update",
        metric="transitions_per_s",
        unit="transitions/s",
        work_per_iter=float(batch_size),
        iters=int(iters_update),
        warmup=0,
        trials=trials,
        params={
            "device": dev_s,
            "batch": int(batch_size),
            "obs_dim": int(obs_dim),
            "n_actions": int(n_actions),
            "phi_dim": int(phi_dim),
            "hidden": int(hidden),
            "region_capacity": int(region_capacity),
            "depth_max": int(depth_max),
            "prefill": int(prefill),
        },
        device=device,
        setup_fn=setup_update,
    )

    return [compute_res, update_res]


class _LinearValue(torch.nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Linear(int(obs_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _bench_gae(
    *,
    base_seed: int,
    device: torch.device,
    trials: int,
    warmup: int,
    iters: int,
    T: int,
    B: int,
    obs_dim: int,
) -> BenchResult:
    from irl.algo.advantage import compute_gae

    name = "gae.compute_gae"
    seed = _stable_seed(base_seed, name)
    _seed_everything(seed)
    rng = np.random.default_rng(seed)

    obs = rng.standard_normal((int(T), int(B), int(obs_dim))).astype(np.float32)
    next_obs = rng.standard_normal((int(T), int(B), int(obs_dim))).astype(np.float32)
    rewards = rng.standard_normal((int(T), int(B))).astype(np.float32)

    terminals = np.zeros((int(T), int(B)), dtype=np.float32)
    terminals[-1, :] = 1.0

    truncations = np.zeros((int(T), int(B)), dtype=np.float32)
    if int(T) >= 3:
        mid = int(T) // 2
        trunc_mask = rng.random((int(B),)) < 0.2
        truncations[mid, trunc_mask] = 1.0

    def setup_fn(trial_idx: int) -> Callable[[], None]:
        _seed_everything(seed + int(trial_idx))
        vf = _LinearValue(int(obs_dim)).to(device)

        batch = {
            "obs": obs,
            "next_observations": next_obs,
            "rewards": rewards,
            "terminals": terminals,
            "truncations": truncations,
        }

        def run_once() -> None:
            _ = compute_gae(
                batch,
                vf,
                gamma=0.99,
                lam=0.95,
                bootstrap_on_timeouts=True,
            )

        return run_once

    return _bench_throughput_with_setup(
        name=name,
        metric="transitions_per_s",
        unit="transitions/s",
        work_per_iter=float(int(T) * int(B)),
        iters=int(iters),
        warmup=warmup,
        trials=trials,
        params={
            "device": str(device),
            "T": int(T),
            "B": int(B),
            "obs_dim": int(obs_dim),
            "gamma": 0.99,
            "lam": 0.95,
        },
        device=device,
        setup_fn=setup_fn,
    )


def _bench_ppo_update(
    *,
    base_seed: int,
    device: torch.device,
    trials: int,
    warmup: int,
    iters: int,
    N: int,
    obs_dim: int,
    n_actions: int,
    epochs: int,
    minibatches: int,
    lr: float,
) -> BenchResult:
    import gymnasium as gym

    from torch.optim import Adam

    from irl.algo.ppo import ppo_update
    from irl.cfg.schema import PPOConfig
    from irl.models.networks import PolicyNetwork, ValueNetwork

    name = "ppo.ppo_update"
    seed = _stable_seed(base_seed, name)
    _seed_everything(seed)
    rng = np.random.default_rng(seed)

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(int(obs_dim),), dtype=np.float32)
    act_space = gym.spaces.Discrete(int(n_actions))

    obs = rng.standard_normal((int(N), int(obs_dim))).astype(np.float32)
    actions = rng.integers(0, int(n_actions), size=(int(N),), endpoint=False, dtype=np.int64)

    adv = torch.as_tensor(rng.standard_normal((int(N),)).astype(np.float32), device=device)
    vt = torch.as_tensor(rng.standard_normal((int(N),)).astype(np.float32), device=device)

    cfg = PPOConfig(
        steps_per_update=int(N),
        minibatches=int(minibatches),
        epochs=int(epochs),
        learning_rate=float(lr),
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.0,
        value_coef=0.5,
        value_clip_range=0.0,
        kl_penalty_coef=0.0,
        kl_stop=0.0,
    )

    def setup_fn(trial_idx: int) -> Callable[[], None]:
        _seed_everything(seed + int(trial_idx))
        policy = PolicyNetwork(obs_space, act_space).to(device)
        value = ValueNetwork(obs_space).to(device)
        pol_opt = Adam(policy.parameters(), lr=float(lr))
        val_opt = Adam(value.parameters(), lr=float(lr))

        batch = {"obs": obs, "actions": actions}

        def run_once() -> None:
            _ = ppo_update(
                policy,
                value,
                batch,
                adv,
                vt,
                cfg,
                optimizers=(pol_opt, val_opt),
                return_stats=False,
            )

        return run_once

    return _bench_throughput_with_setup(
        name=name,
        metric="samples_per_s",
        unit="samples/s",
        work_per_iter=float(N),
        iters=int(iters),
        warmup=warmup,
        trials=trials,
        params={
            "device": str(device),
            "N": int(N),
            "obs_dim": int(obs_dim),
            "n_actions": int(n_actions),
            "epochs": int(epochs),
            "minibatches": int(minibatches),
            "lr": float(lr),
            "clip_range": 0.2,
        },
        device=device,
        setup_fn=setup_fn,
    )


def _bench_env_step(
    *,
    base_seed: int,
    trials: int,
    warmup: int,
    steps: int,
    env_id: str,
    vec_envs: int,
    frame_skip: int,
) -> BenchResult:
    from irl.envs.builder import make_env

    name = f"env.step.sync_vec{int(vec_envs)}"
    seed = _stable_seed(base_seed, name)
    rng = np.random.default_rng(seed)

    def setup_fn(trial_idx: int) -> Callable[[], None]:
        trial_seed = int(seed) + 17 * int(trial_idx)
        env = make_env(
            env_id=str(env_id),
            num_envs=int(vec_envs),
            seed=int(trial_seed),
            frame_skip=int(frame_skip),
            domain_randomization=False,
            discrete_actions=True,
            car_action_set=None,
            render_mode=None,
            async_vector=False,
            deterministic=True,
            make_kwargs=None,
        )

        act_space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)

        B = int(getattr(env, "num_envs", 1))
        if B <= 1:
            obs, _ = env.reset(seed=int(trial_seed))

            def run_once() -> None:
                nonlocal obs
                done_count = 0
                for _ in range(int(steps)):
                    a = int(rng.integers(0, int(getattr(act_space, "n", 2)), endpoint=False))
                    obs, _r, term, trunc, _info = env.step(a)
                    if bool(term) or bool(trunc):
                        done_count += 1
                        obs, _ = env.reset(seed=int(trial_seed) + done_count)

            def run_and_close() -> None:
                try:
                    run_once()
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass

            return run_and_close

        try:
            _ = env.reset(seed=[int(trial_seed) + i for i in range(B)])
        except Exception:
            _ = env.reset(seed=int(trial_seed))

        def run_once() -> None:
            try:
                n = int(getattr(act_space, "n", 2))
            except Exception:
                n = 2
            for _ in range(int(steps)):
                a = rng.integers(0, int(n), size=(B,), endpoint=False, dtype=np.int64)
                _ = env.step(a)

        def run_and_close() -> None:
            try:
                run_once()
            finally:
                try:
                    env.close()
                except Exception:
                    pass

        return run_and_close

    transitions = float(int(steps) * int(vec_envs))
    return _bench_throughput_with_setup(
        name=name,
        metric="transitions_per_s",
        unit="transitions/s",
        work_per_iter=transitions,
        iters=1,
        warmup=warmup,
        trials=trials,
        params={
            "env_id": str(env_id),
            "vec_envs": int(vec_envs),
            "frame_skip": int(frame_skip),
            "steps_per_trial": int(steps),
            "async_vector": False,
        },
        device=torch.device("cpu"),
        setup_fn=setup_fn,
    )


def _bench_glpe_gate_median_cache(
    *,
    base_seed: int,
    device: torch.device,
    trials: int,
    warmup: int,
    iters: int,
    batch_size: int,
    obs_dim: int,
    n_actions: int,
    phi_dim: int,
    hidden: int,
    region_capacity: int,
    depth_max: int,
    prefill: int,
    cache_interval: int,
) -> list[BenchResult]:
    import gymnasium as gym

    from irl.intrinsic.glpe import GLPE
    from irl.intrinsic.icm import ICMConfig

    name_a = "glpe.gate_median_cache.baseline"
    name_b = "glpe.gate_median_cache.cached"
    name_s = "glpe.gate_median_cache.speedup"

    seed = _stable_seed(base_seed, "glpe.gate_median_cache")
    _seed_everything(seed)
    rng = np.random.default_rng(seed)

    dev_s = str(device)

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(int(obs_dim),), dtype=np.float32)
    act_space = gym.spaces.Discrete(int(n_actions))
    icm_cfg = ICMConfig(phi_dim=int(phi_dim), hidden=(int(hidden), int(hidden)))

    pre_mod = GLPE(
        obs_space,
        act_space,
        device=dev_s,
        icm_cfg=icm_cfg,
        region_capacity=int(region_capacity),
        depth_max=int(depth_max),
        normalize_inside=True,
        gating_enabled=True,
        checkpoint_include_points=False,
    )

    _prefill_intrinsic(
        pre_mod,
        rng,
        transitions=int(prefill),
        batch=int(max(1, batch_size)),
        obs_dim=int(obs_dim),
        n_actions=int(n_actions),
    )

    prefilled_sd = pre_mod.state_dict()

    obs, next_obs, actions = _make_vec_batch(
        rng, n=int(batch_size), obs_dim=int(obs_dim), n_actions=int(n_actions)
    )

    base_vals: list[float] = []
    base_durs: list[float] = []
    cached_vals: list[float] = []
    cached_durs: list[float] = []
    speedups: list[float] = []

    sync = (lambda: _maybe_cuda_sync(device)) if device.type == "cuda" else None

    for trial_idx in range(int(max(1, trials))):
        _seed_everything(seed + 100 * int(trial_idx))

        mod_a = GLPE(
            obs_space,
            act_space,
            device=dev_s,
            icm_cfg=icm_cfg,
            region_capacity=int(region_capacity),
            depth_max=int(depth_max),
            normalize_inside=True,
            gating_enabled=True,
            checkpoint_include_points=False,
        )
        mod_a.load_state_dict(prefilled_sd, strict=True)
        if hasattr(mod_a, "gate_median_cache_interval"):
            mod_a.gate_median_cache_interval = 1

        mod_b = GLPE(
            obs_space,
            act_space,
            device=dev_s,
            icm_cfg=icm_cfg,
            region_capacity=int(region_capacity),
            depth_max=int(depth_max),
            normalize_inside=True,
            gating_enabled=True,
            checkpoint_include_points=False,
        )
        mod_b.load_state_dict(prefilled_sd, strict=True)
        if hasattr(mod_b, "gate_median_cache_interval"):
            mod_b.gate_median_cache_interval = int(max(1, cache_interval))

        def run_a() -> None:
            _ = mod_a.compute_batch(obs, next_obs, actions, reduction="none")

        def run_b() -> None:
            _ = mod_b.compute_batch(obs, next_obs, actions, reduction="none")

        for _ in range(int(max(0, warmup))):
            run_a()
        dt_a = _time_loop(run_a, iters=int(iters), sync=sync)
        tps_a = (float(batch_size) * float(iters)) / max(1e-12, float(dt_a))
        base_durs.append(float(dt_a))
        base_vals.append(float(tps_a))

        for _ in range(int(max(0, warmup))):
            run_b()
        dt_b = _time_loop(run_b, iters=int(iters), sync=sync)
        tps_b = (float(batch_size) * float(iters)) / max(1e-12, float(dt_b))
        cached_durs.append(float(dt_b))
        cached_vals.append(float(tps_b))

        speedups.append(float(tps_b) / max(1e-12, float(tps_a)))

    baseline = BenchResult(
        name=name_a,
        metric="transitions_per_s",
        unit="transitions/s",
        params={
            "device": dev_s,
            "batch": int(batch_size),
            "obs_dim": int(obs_dim),
            "n_actions": int(n_actions),
            "phi_dim": int(phi_dim),
            "hidden": int(hidden),
            "region_capacity": int(region_capacity),
            "depth_max": int(depth_max),
            "prefill": int(prefill),
            "cache_interval": 1,
            "iters": int(iters),
        },
        values=base_vals,
        durations_s=base_durs,
        error=None,
    )

    cached = BenchResult(
        name=name_b,
        metric="transitions_per_s",
        unit="transitions/s",
        params={
            "device": dev_s,
            "batch": int(batch_size),
            "obs_dim": int(obs_dim),
            "n_actions": int(n_actions),
            "phi_dim": int(phi_dim),
            "hidden": int(hidden),
            "region_capacity": int(region_capacity),
            "depth_max": int(depth_max),
            "prefill": int(prefill),
            "cache_interval": int(max(1, cache_interval)),
            "iters": int(iters),
        },
        values=cached_vals,
        durations_s=cached_durs,
        error=None,
    )

    speed = BenchResult(
        name=name_s,
        metric="speedup",
        unit="x",
        params={
            "device": dev_s,
            "baseline_cache_interval": 1,
            "cached_cache_interval": int(max(1, cache_interval)),
        },
        values=speedups,
        durations_s=[float("nan")] * len(speedups),
        error=None,
    )

    return [baseline, cached, speed]


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
    quick: bool = True,
) -> dict[str, Any]:
    dev = torch.device(str(device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is False.")

    prev_threads = int(torch.get_num_threads())
    torch.set_num_threads(int(max(1, threads)))

    t0 = time.perf_counter()
    try:
        meta = _system_info(device=dev, seed=int(seed))

        trials = 3 if bool(quick) else 7
        warmup = 1 if bool(quick) else 2

        results: list[BenchResult] = []

        try:
            results.append(
                _bench_kdtree_bulk_insert(
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
                _bench_glpe_pipeline(
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
                _bench_riac_pipeline(
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
                _bench_gae(
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
                _bench_ppo_update(
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
                _bench_env_step(
                    base_seed=int(seed),
                    trials=trials,
                    warmup=0,
                    steps=500 if bool(quick) else 1500,
                    env_id="MountainCar-v0",
                    vec_envs=1,
                    frame_skip=1,
                )
            )
            results.append(
                _bench_env_step(
                    base_seed=int(seed),
                    trials=trials,
                    warmup=0,
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
                _bench_glpe_gate_median_cache(
                    base_seed=int(seed),
                    device=dev,
                    trials=trials,
                    warmup=1,
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
