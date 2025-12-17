from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
import torch

from irl.intrinsic.glpe import GLPE
from irl.intrinsic.icm import ICMConfig


def _make_batch(rng: np.random.Generator, *, n: int, obs_dim: int, n_actions: int):
    obs = rng.standard_normal((n, obs_dim)).astype(np.float32)
    next_obs = rng.standard_normal((n, obs_dim)).astype(np.float32)
    actions = rng.integers(0, n_actions, size=(n,), endpoint=False, dtype=np.int64)
    return obs, next_obs, actions


def _time_compute_batch(mod: GLPE, obs, next_obs, actions, *, iters: int, warmup: int) -> float:
    for _ in range(int(warmup)):
        _ = mod.compute_batch(obs, next_obs, actions, reduction="none")

    t0 = time.perf_counter()
    for _ in range(int(iters)):
        _ = mod.compute_batch(obs, next_obs, actions, reduction="none")
    t1 = time.perf_counter()
    return float(t1 - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--obs-dim", type=int, default=8)
    ap.add_argument("--actions", type=int, default=6)
    ap.add_argument("--phi-dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--region-capacity", type=int, default=32)
    ap.add_argument("--depth-max", type=int, default=10)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cache-interval", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(int(max(1, args.threads)))
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(int(args.obs_dim),), dtype=np.float32)
    act_space = gym.spaces.Discrete(int(args.actions))

    icm_cfg = ICMConfig(phi_dim=int(args.phi_dim), hidden=(int(args.hidden), int(args.hidden)))

    mod_a = GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=int(args.region_capacity),
        depth_max=int(args.depth_max),
        normalize_inside=True,
        gating_enabled=True,
    )
    mod_b = GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=int(args.region_capacity),
        depth_max=int(args.depth_max),
        normalize_inside=True,
        gating_enabled=True,
    )
    mod_b.load_state_dict(mod_a.state_dict(), strict=True)

    obs, next_obs, actions = _make_batch(
        rng, n=int(args.batch), obs_dim=int(args.obs_dim), n_actions=int(args.actions)
    )

    if hasattr(mod_a, "gate_median_cache_interval"):
        mod_a.gate_median_cache_interval = 1
    t_a = _time_compute_batch(
        mod_a, obs, next_obs, actions, iters=int(args.iters), warmup=int(args.warmup)
    )
    n_trans = int(args.batch) * int(args.iters)
    sps_a = float(n_trans) / max(1e-9, t_a)

    print(f"[baseline] cache_interval=1   time={t_a:.3f}s  transitions/s={sps_a:.1f}")

    if not hasattr(mod_b, "gate_median_cache_interval"):
        print("[cached]   gate_median_cache_interval not available in this build")
        return

    mod_b.gate_median_cache_interval = 1
    _ = mod_b.compute_batch(obs, next_obs, actions, reduction="none")

    mod_b.gate_median_cache_interval = int(max(1, args.cache_interval))
    t_b = _time_compute_batch(
        mod_b, obs, next_obs, actions, iters=int(args.iters), warmup=int(args.warmup)
    )
    sps_b = float(n_trans) / max(1e-9, t_b)

    speedup = (sps_b / sps_a) if sps_a > 0 else float("nan")
    print(
        f"[cached]   cache_interval={int(args.cache_interval)} "
        f"time={t_b:.3f}s  transitions/s={sps_b:.1f}  speedup={speedup:.3f}x"
    )


if __name__ == "__main__":
    main()
