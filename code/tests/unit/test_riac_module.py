import gymnasium as gym
import numpy as np

from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.riac import RIAC, simulate_lp_emas


def test_simulate_lp_emas_produces_nonnegative_lp():
    errors = np.linspace(1.0, 0.0, num=200)
    _, _, lp = simulate_lp_emas(errors, beta_long=0.995, beta_short=0.90)
    lp_arr = np.asarray(lp, dtype=np.float64)
    assert np.all(lp_arr >= 0.0)
    assert lp_arr[-1] > 0.0


def test_riac_export_diagnostics_creates_files(tmp_path):
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)
    riac = RIAC(obs_space, act_space, device="cpu", icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)))
    assert bool(getattr(riac, "outputs_normalized", False))

    o = np.zeros((4, 3), dtype=np.float32)
    op = np.ones((4, 3), dtype=np.float32)
    a = np.zeros((4,), dtype=np.int64)

    _ = riac.compute_batch(o, op, a)

    out_dir = tmp_path / "diagnostics"
    riac.export_diagnostics(out_dir, step=1000)

    regions = out_dir / "regions.jsonl"
    gates = out_dir / "gates.csv"
    assert regions.exists() and regions.stat().st_size > 0
    assert gates.exists() and gates.stat().st_size > 0
