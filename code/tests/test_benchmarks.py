from __future__ import annotations

import gymnasium as gym
import numpy as np

from irl.intrinsic.glpe import GLPE
from irl.intrinsic.icm import ICMConfig


def test_glpe_restore_with_points_keeps_store_depth_max() -> None:
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    depth_max = 4
    phi_dim = 8
    mod = GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=phi_dim, hidden=(16, 16)),
        region_capacity=4,
        depth_max=depth_max,
        normalize_inside=True,
        gating_enabled=False,
        checkpoint_include_points=True,
    )

    rng = np.random.default_rng(0)
    pts = rng.standard_normal((200, int(mod.phi_dim))).astype(np.float32)
    _ = mod.store.bulk_insert(pts)

    assert int(mod.store.num_regions()) > 1
    assert int(mod.store.depth_max) == int(depth_max)

    sd = mod.state_dict()

    restored = GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=phi_dim, hidden=(16, 16)),
        region_capacity=4,
        depth_max=depth_max,
        normalize_inside=True,
        gating_enabled=False,
        checkpoint_include_points=True,
    )
    restored.load_state_dict(sd, strict=True)

    assert int(restored.store.depth_max) == int(depth_max)
    assert int(restored.store.num_regions()) == int(mod.store.num_regions())
