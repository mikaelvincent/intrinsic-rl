from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from irl.intrinsic.glpe import GLPE
from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.riac import RIAC


def _fill_store(mod: object, *, n_points: int, seed: int) -> None:
    rng = np.random.default_rng(int(seed))
    dim = int(getattr(mod, "phi_dim"))
    pts = rng.standard_normal((int(n_points), dim)).astype(np.float32)
    store = getattr(mod, "store")
    for p in pts:
        store.insert(p)


def _save_state_dict(sd: dict, path: Path) -> int:
    torch.save(sd, path)
    return int(path.stat().st_size)


def _torch_load_any(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_glpe_state_dict_can_omit_kdtree_points(tmp_path: Path) -> None:
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)
    icm_cfg = ICMConfig(phi_dim=32, hidden=(32, 32))

    mod = GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=100_000,
        depth_max=12,
        normalize_inside=False,
        gating_enabled=False,
        checkpoint_include_points=True,
    )
    _fill_store(mod, n_points=5000, seed=0)

    p_with = tmp_path / "glpe_with_points.pt"
    size_with = _save_state_dict(mod.state_dict(), p_with)

    mod.checkpoint_include_points = False
    p_without = tmp_path / "glpe_without_points.pt"
    size_without = _save_state_dict(mod.state_dict(), p_without)

    assert size_without < size_with
    assert size_without <= int(size_with * 0.4)

    mod2 = GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=100_000,
        depth_max=12,
        normalize_inside=False,
        gating_enabled=False,
        checkpoint_include_points=False,
    )
    sd_no_points = _torch_load_any(p_without)
    mod2.load_state_dict(sd_no_points, strict=True)

    assert int(mod2.store.num_regions()) == int(mod.store.num_regions())
    assert int(mod2.store.depth_max) == 0


def test_riac_state_dict_can_omit_kdtree_points(tmp_path: Path) -> None:
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)
    icm_cfg = ICMConfig(phi_dim=32, hidden=(32, 32))

    mod = RIAC(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=100_000,
        depth_max=12,
        checkpoint_include_points=True,
    )
    _fill_store(mod, n_points=5000, seed=1)

    p_with = tmp_path / "riac_with_points.pt"
    size_with = _save_state_dict(mod.state_dict(), p_with)

    mod.checkpoint_include_points = False
    p_without = tmp_path / "riac_without_points.pt"
    size_without = _save_state_dict(mod.state_dict(), p_without)

    assert size_without < size_with
    assert size_without <= int(size_with * 0.4)

    mod2 = RIAC(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=100_000,
        depth_max=12,
        checkpoint_include_points=False,
    )
    sd_no_points = _torch_load_any(p_without)
    mod2.load_state_dict(sd_no_points, strict=True)

    assert int(mod2.store.num_regions()) == int(mod.store.num_regions())
    assert int(mod2.store.depth_max) == 0
