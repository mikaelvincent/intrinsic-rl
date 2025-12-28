from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from .core import (
    BenchResult,
    _bench_throughput_with_setup,
    _maybe_cuda_sync,
    _seed_everything,
    _stable_seed,
    _time_loop,
)


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


def _prefill_intrinsic(
    mod: object, rng: np.random.Generator, *, transitions: int, batch: int, obs_dim: int, n_actions: int
):
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
        rollout_steps_per_env=int(N),
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
        checkpoint_include_points=True,
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
            checkpoint_include_points=True,
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
            checkpoint_include_points=True,
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
