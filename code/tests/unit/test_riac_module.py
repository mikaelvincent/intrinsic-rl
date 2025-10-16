import numpy as np
import gymnasium as gym
import torch
from pathlib import Path
from tempfile import TemporaryDirectory

from irl.intrinsic.riac import RIAC
from irl.intrinsic.icm import ICMConfig
from irl.intrinsic import IntrinsicOutput


def _rand_batch(obs_dim, act_space, B=8, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((B, obs_dim)).astype(np.float32)
    next_obs = rng.standard_normal((B, obs_dim)).astype(np.float32)

    if isinstance(act_space, gym.spaces.Discrete):
        acts = rng.integers(0, act_space.n, size=(B,), endpoint=False, dtype=np.int64)
    else:
        low = np.where(np.isfinite(act_space.low), act_space.low, -1.0)
        high = np.where(np.isfinite(act_space.high), act_space.high, 1.0)
        acts = rng.uniform(low, high, size=(B, act_space.shape[0])).astype(np.float32)
    return obs, next_obs, acts


def test_riac_shapes_and_update():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    # Small nets for speed
    riac = RIAC(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=32, hidden=(64, 64)),
        region_capacity=20,
        depth_max=4,
        alpha_lp=0.5,
    )

    obs, next_obs, acts = _rand_batch(5, act_space, B=12, seed=1)

    # Vectorized compute
    r = riac.compute_batch(obs, next_obs, acts)  # [B]
    assert r.shape == (12,)
    assert torch.isfinite(r).all()

    # Loss/Update (ICM training path)
    losses = riac.loss(obs, next_obs, acts)
    for k in ["total", "icm_forward", "icm_inverse", "intrinsic_mean"]:
        assert k in losses and torch.isfinite(losses[k])

    metrics = riac.update(obs, next_obs, acts, steps=2)
    for v in metrics.values():
        assert np.isfinite(v)

    # Single transition compute
    tr = type("T", (), {"s": obs[0], "a": int(acts[0]), "r_ext": 0.0, "s_next": next_obs[0]})
    out = riac.compute(tr)
    assert isinstance(out, IntrinsicOutput)
    assert np.isfinite(out.r_int)


def test_riac_export_diagnostics_creates_files():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)
    riac = RIAC(obs_space, act_space, device="cpu", icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)))

    # Generate a few transitions to populate regions/stats
    o = np.zeros((4, 3), dtype=np.float32)
    op = np.ones((4, 3), dtype=np.float32)
    a = np.zeros((4,), dtype=np.int64)
    _ = riac.compute_batch(o, op, a)  # update region stats internally

    with TemporaryDirectory() as td:
        out_dir = Path(td)
        riac.export_diagnostics(out_dir, step=1000)
        assert (out_dir / "regions.jsonl").exists()
        assert (out_dir / "gates.csv").exists()
        # files should be non-empty
        assert (out_dir / "regions.jsonl").stat().st_size > 0
        assert (out_dir / "gates.csv").stat().st_size > 0


def test_riac_outputs_normalized_flag():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)
    riac = RIAC(obs_space, act_space, device="cpu")
    assert hasattr(riac, "outputs_normalized") and bool(riac.outputs_normalized)
