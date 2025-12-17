from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import pytest
import torch

from irl.intrinsic.glpe import GLPE
from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.riac import RIAC
from irl.utils.checkpoint import CheckpointManager


def _payload(step: int) -> dict:
    return {"step": int(step), "meta": {"note": "test"}}


def test_checkpoint_prune_preserves_step0() -> None:
    with TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        cm = CheckpointManager(run_dir, interval_steps=10, max_to_keep=2)

        for step in (0, 10, 20, 30):
            cm.save(step=step, payload=_payload(step))

        ckpt_dir = run_dir / "checkpoints"
        kept = sorted(p.name for p in ckpt_dir.glob("ckpt_step_*.pt"))

        assert "ckpt_step_0.pt" in kept
        assert "ckpt_step_30.pt" in kept
        assert "ckpt_step_20.pt" in kept
        assert "ckpt_step_10.pt" not in kept


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


def _make_spaces() -> tuple[gym.Space, gym.Space]:
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)
    return obs_space, act_space


def _make_glpe(obs_space: gym.Space, act_space: gym.Space, *, include_points: bool) -> GLPE:
    icm_cfg = ICMConfig(phi_dim=32, hidden=(32, 32))
    return GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=100_000,
        depth_max=12,
        normalize_inside=False,
        gating_enabled=False,
        checkpoint_include_points=bool(include_points),
    )


def _make_riac(obs_space: gym.Space, act_space: gym.Space, *, include_points: bool) -> RIAC:
    icm_cfg = ICMConfig(phi_dim=32, hidden=(32, 32))
    return RIAC(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=100_000,
        depth_max=12,
        checkpoint_include_points=bool(include_points),
    )


@pytest.mark.parametrize(
    "make_module, method_tag",
    [(_make_glpe, "glpe"), (_make_riac, "riac")],
)
def test_state_dict_can_omit_kdtree_points(tmp_path: Path, make_module, method_tag: str) -> None:
    obs_space, act_space = _make_spaces()
    mod = make_module(obs_space, act_space, include_points=True)
    _fill_store(mod, n_points=5000, seed=0)

    p_with = tmp_path / f"{method_tag}_with_points.pt"
    size_with = _save_state_dict(mod.state_dict(), p_with)

    mod.checkpoint_include_points = False
    p_without = tmp_path / f"{method_tag}_without_points.pt"
    size_without = _save_state_dict(mod.state_dict(), p_without)

    assert size_without < size_with
    assert size_without <= int(size_with * 0.4)

    mod2 = make_module(obs_space, act_space, include_points=False)
    sd_no_points = _torch_load_any(p_without)
    mod2.load_state_dict(sd_no_points, strict=True)

    assert int(mod2.store.num_regions()) == int(mod.store.num_regions())
    assert int(mod2.store.depth_max) == 0
