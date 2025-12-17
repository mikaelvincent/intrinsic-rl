from __future__ import annotations

import argparse
import platform
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg.schema import PPOConfig
from irl.intrinsic.glpe import GLPE
from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.regions import KDTreeRegionStore
from irl.intrinsic.riac import RIAC
from irl.models.networks import PolicyNetwork, ValueNetwork


def _time_loop(fn, *, iters: int, warmup: int) -> float:
    for _ in range(int(max(0, warmup))):
        fn()
    t0 = time.perf_counter()
    for _ in range(int(max(1, iters))):
        fn()
    t1 = time.perf_counter()
    return float(t1 - t0)


class _LinearValue(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(int(obs_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _print_env(device: str) -> None:
    print(
        "env:",
        f"python={sys.version.split()[0]}",
        f"torch={torch.__version__}",
        f"platform={platform.platform()}",
        f"device={device}",
        f"threads={torch.get_num_threads()}",
    )


def _bench_kdtree(rng: np.random.Generator, *, n: int, dim: int, cap: int, depth: int, iters: int, warmup: int):
    pts = rng.standard_normal((int(n), int(dim))).astype(np.float32)

    def run_once() -> None:
        store = KDTreeRegionStore(dim=int(dim), capacity=int(cap), depth_max=int(depth))
        _ = store.bulk_insert(pts)

    dt = _time_loop(run_once, iters=iters, warmup=warmup)
    sps = (float(n) * float(max(1, iters))) / max(1e-9, dt)
    print(f"[kdtree] bulk_insert  time={dt:.3f}s  points/s={sps:.1f}")


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
        _ = mod.compute_batch(obs, next_obs, actions, reduction="none")
        remaining -= n


def _bench_glpe(
    rng: np.random.Generator,
    *,
    device: str,
    n: int,
    obs_dim: int,
    n_actions: int,
    phi_dim: int,
    hidden: int,
    region_capacity: int,
    depth_max: int,
    gate_cache_interval: int,
    prefill: int,
    iters: int,
    warmup: int,
):
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(int(obs_dim),), dtype=np.float32)
    act_space = gym.spaces.Discrete(int(n_actions))
    icm_cfg = ICMConfig(phi_dim=int(phi_dim), hidden=(int(hidden), int(hidden)))

    mod = GLPE(
        obs_space,
        act_space,
        device=str(device),
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
        batch=int(n),
        obs_dim=int(obs_dim),
        n_actions=int(n_actions),
    )

    obs, next_obs, actions = _make_vec_batch(rng, n=int(n), obs_dim=int(obs_dim), n_actions=int(n_actions))

    def run_compute() -> None:
        _ = mod.compute_batch(obs, next_obs, actions, reduction="none")

    def run_update() -> None:
        _ = mod.update(obs, next_obs, actions, steps=1)

    dt_c = _time_loop(run_compute, iters=iters, warmup=warmup)
    sps_c = (float(n) * float(max(1, iters))) / max(1e-9, dt_c)
    print(f"[glpe]  compute_batch time={dt_c:.3f}s  transitions/s={sps_c:.1f}")

    dt_u = _time_loop(run_update, iters=iters, warmup=0)
    sps_u = (float(n) * float(max(1, iters))) / max(1e-9, dt_u)
    print(f"[glpe]  update(steps=1) time={dt_u:.3f}s  transitions/s={sps_u:.1f}")


def _bench_riac(
    rng: np.random.Generator,
    *,
    device: str,
    n: int,
    obs_dim: int,
    n_actions: int,
    phi_dim: int,
    hidden: int,
    region_capacity: int,
    depth_max: int,
    prefill: int,
    iters: int,
    warmup: int,
):
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(int(obs_dim),), dtype=np.float32)
    act_space = gym.spaces.Discrete(int(n_actions))
    icm_cfg = ICMConfig(phi_dim=int(phi_dim), hidden=(int(hidden), int(hidden)))

    mod = RIAC(
        obs_space,
        act_space,
        device=str(device),
        icm_cfg=icm_cfg,
        region_capacity=int(region_capacity),
        depth_max=int(depth_max),
    )

    _prefill_intrinsic(
        mod,
        rng,
        transitions=int(prefill),
        batch=int(n),
        obs_dim=int(obs_dim),
        n_actions=int(n_actions),
    )

    obs, next_obs, actions = _make_vec_batch(rng, n=int(n), obs_dim=int(obs_dim), n_actions=int(n_actions))

    def run_compute() -> None:
        _ = mod.compute_batch(obs, next_obs, actions, reduction="none")

    def run_update() -> None:
        _ = mod.update(obs, next_obs, actions, steps=1)

    dt_c = _time_loop(run_compute, iters=iters, warmup=warmup)
    sps_c = (float(n) * float(max(1, iters))) / max(1e-9, dt_c)
    print(f"[riac]  compute_batch time={dt_c:.3f}s  transitions/s={sps_c:.1f}")

    dt_u = _time_loop(run_update, iters=iters, warmup=0)
    sps_u = (float(n) * float(max(1, iters))) / max(1e-9, dt_u)
    print(f"[riac]  update(steps=1) time={dt_u:.3f}s  transitions/s={sps_u:.1f}")


def _bench_gae(
    rng: np.random.Generator,
    *,
    device: str,
    T: int,
    B: int,
    obs_dim: int,
    iters: int,
    warmup: int,
):
    dev = torch.device(str(device))
    obs = torch.as_tensor(
        rng.standard_normal((int(T), int(B), int(obs_dim))).astype(np.float32),
        device=dev,
    )
    next_obs = torch.as_tensor(
        rng.standard_normal((int(T), int(B), int(obs_dim))).astype(np.float32),
        device=dev,
    )
    rewards = torch.as_tensor(
        rng.standard_normal((int(T), int(B))).astype(np.float32),
        device=dev,
    )

    terminals = torch.zeros((int(T), int(B)), dtype=torch.float32, device=dev)
    terminals[-1] = 1.0

    truncations = torch.zeros((int(T), int(B)), dtype=torch.float32, device=dev)
    if int(T) >= 3:
        mid = int(T) // 2
        trunc_mask = rng.random((int(B),)) < 0.2
        truncations[mid, torch.as_tensor(trunc_mask, device=dev)] = 1.0

    vf = _LinearValue(int(obs_dim)).to(dev)

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

    N = int(T) * int(B)
    dt = _time_loop(run_once, iters=iters, warmup=warmup)
    sps = (float(N) * float(max(1, iters))) / max(1e-9, dt)
    print(f"[gae]   compute_gae   time={dt:.3f}s  transitions/s={sps:.1f}")


def _bench_ppo(
    rng: np.random.Generator,
    *,
    device: str,
    N: int,
    obs_dim: int,
    n_actions: int,
    epochs: int,
    minibatches: int,
    lr: float,
    iters: int,
    warmup: int,
):
    dev = torch.device(str(device))
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(int(obs_dim),), dtype=np.float32)
    act_space = gym.spaces.Discrete(int(n_actions))

    policy = PolicyNetwork(obs_space, act_space).to(dev)
    value = ValueNetwork(obs_space).to(dev)

    pol_opt = Adam(policy.parameters(), lr=float(lr))
    val_opt = Adam(value.parameters(), lr=float(lr))

    obs = rng.standard_normal((int(N), int(obs_dim))).astype(np.float32)
    actions = rng.integers(0, int(n_actions), size=(int(N),), endpoint=False, dtype=np.int64)
    adv = rng.standard_normal((int(N),)).astype(np.float32)
    vt = rng.standard_normal((int(N),)).astype(np.float32)

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

    dt = _time_loop(run_once, iters=iters, warmup=warmup)
    sps = (float(N) * float(max(1, iters))) / max(1e-9, dt)
    print(f"[ppo]   ppo_update    time={dt:.3f}s  samples/s={sps:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)

    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--obs-dim", type=int, default=8)
    ap.add_argument("--actions", type=int, default=6)
    ap.add_argument("--phi-dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--region-capacity", type=int, default=32)
    ap.add_argument("--depth-max", type=int, default=10)
    ap.add_argument("--gate-cache-interval", type=int, default=64)
    ap.add_argument("--prefill", type=int, default=0)

    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)

    ap.add_argument("--gae-T", type=int, default=128)
    ap.add_argument("--gae-B", type=int, default=16)

    ap.add_argument("--ppo-N", type=int, default=8192)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--ppo-minibatches", type=int, default=8)
    ap.add_argument("--ppo-lr", type=float, default=3e-4)

    args = ap.parse_args()

    torch.set_num_threads(int(max(1, args.threads)))
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    _print_env(str(args.device))

    _bench_kdtree(
        rng,
        n=int(args.batch),
        dim=int(args.phi_dim),
        cap=int(args.region_capacity),
        depth=int(args.depth_max),
        iters=int(args.iters),
        warmup=int(args.warmup),
    )

    _bench_glpe(
        rng,
        device=str(args.device),
        n=int(args.batch),
        obs_dim=int(args.obs_dim),
        n_actions=int(args.actions),
        phi_dim=int(args.phi_dim),
        hidden=int(args.hidden),
        region_capacity=int(args.region_capacity),
        depth_max=int(args.depth_max),
        gate_cache_interval=int(args.gate_cache_interval),
        prefill=int(args.prefill),
        iters=int(args.iters),
        warmup=int(args.warmup),
    )

    _bench_riac(
        rng,
        device=str(args.device),
        n=int(args.batch),
        obs_dim=int(args.obs_dim),
        n_actions=int(args.actions),
        phi_dim=int(args.phi_dim),
        hidden=int(args.hidden),
        region_capacity=int(args.region_capacity),
        depth_max=int(args.depth_max),
        prefill=int(args.prefill),
        iters=int(args.iters),
        warmup=int(args.warmup),
    )

    _bench_gae(
        rng,
        device=str(args.device),
        T=int(args.gae_T),
        B=int(args.gae_B),
        obs_dim=int(args.obs_dim),
        iters=int(args.iters),
        warmup=int(args.warmup),
    )

    _bench_ppo(
        rng,
        device=str(args.device),
        N=int(args.ppo_N),
        obs_dim=int(args.obs_dim),
        n_actions=int(args.actions),
        epochs=int(args.ppo_epochs),
        minibatches=int(args.ppo_minibatches),
        lr=float(args.ppo_lr),
        iters=int(args.iters),
        warmup=int(args.warmup),
    )


if __name__ == "__main__":
    main()
