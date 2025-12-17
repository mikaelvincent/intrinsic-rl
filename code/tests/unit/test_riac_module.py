from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.riac import RIAC


def test_riac_export_diagnostics_writes_minimal_records(tmp_path: Path):
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)
    riac = RIAC(obs_space, act_space, device="cpu", icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)))

    o = np.zeros((4, 3), dtype=np.float32)
    op = np.ones((4, 3), dtype=np.float32)
    a = np.zeros((4,), dtype=np.int64)
    _ = riac.compute_batch(o, op, a)

    out_dir = tmp_path / "diagnostics"
    riac.export_diagnostics(out_dir, step=1000)

    regions_path = out_dir / "regions.jsonl"
    gates_path = out_dir / "gates.csv"
    assert regions_path.exists()
    assert gates_path.exists()

    rec0 = json.loads(regions_path.read_text(encoding="utf-8").splitlines()[0])
    assert rec0["step"] == 1000
    assert int(rec0["region_id"]) >= 0
    assert "lp" in rec0
    assert "gate" in rec0
