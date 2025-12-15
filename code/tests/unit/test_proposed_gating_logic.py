import gymnasium as gym

from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.proposed import Proposed, _RegionStats


def test_proposed_gating_off_then_hysteresis_reenable():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float)
    act_space = gym.spaces.Discrete(3)
    mod = Proposed(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)),
        gate_tau_lp_mult=0.5,
        gate_tau_s=0.5,
        gate_min_consec_to_gate=3,
        gate_hysteresis_up_mult=1.1,
        gate_min_regions_for_gating=3,
    )

    mod._stats = {
        0: _RegionStats(ema_long=10.0, ema_short=10.0, count=10, gate=1),
        1: _RegionStats(ema_long=2.0, ema_short=1.0, count=10, gate=1),
        2: _RegionStats(ema_long=2.0, ema_short=1.5, count=10, gate=1),
    }

    for _ in range(3):
        assert mod._maybe_update_gate(0, lp_i=0.0) in (0, 1)
    assert mod._stats[0].gate == 0

    for _ in range(2):
        assert mod._maybe_update_gate(0, lp_i=1.0) in (0, 1)
    assert mod._stats[0].gate == 1
