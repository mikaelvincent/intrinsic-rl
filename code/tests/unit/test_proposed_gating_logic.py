import gymnasium as gym

from irl.intrinsic.proposed import Proposed, _RegionStats  # internal container
from irl.intrinsic.icm import ICMConfig


def test_proposed_gating_off_then_hysteresis_reenable():
    # Small vector spaces for speed
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float)
    act_space = gym.spaces.Discrete(3)

    # Configure thresholds to make the test concise:
    # - Larger tau_lp_mult => larger τ_LP (easier to be "too low")
    # - Smaller tau_s => easier to be "too stochastic"
    # - Fewer consecutive bad updates to trigger gating
    # - Modest hysteresis_up_mult so re-enable is achievable
    mod = Proposed(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)),
        gate_tau_lp_mult=0.5,
        gate_tau_s=0.5,
        gate_min_consec_to_gate=3,
        gate_hysteresis_up_mult=1.1,
    )

    # Seed region stats directly to control medians used by gating.
    # We need >=3 regions with count>0 for medians to engage.
    # Region 0 will be our target for gating.
    mod._stats = {
        0: _RegionStats(ema_long=10.0, ema_short=10.0, count=10, gate=1, bad_consec=0, good_consec=0),  # LP=0, high error -> "bad"
        1: _RegionStats(ema_long=2.0, ema_short=1.0, count=10, gate=1, bad_consec=0, good_consec=0),    # LP=1.0
        2: _RegionStats(ema_long=2.0, ema_short=1.5, count=10, gate=1, bad_consec=0, good_consec=0),    # LP=0.5
    }

    # --- Phase 1: drive gate OFF with consecutive "bad" updates ---
    # For region 0: lp_i = 0.0 (below τ_LP) and S_i large (ema_short high vs median)
    for _ in range(3):
        g = mod._maybe_update_gate(0, lp_i=0.0)
        assert g in (0, 1)
    assert mod._stats[0].gate == 0, "Region should be gated-off after K bad refreshes"

    # --- Phase 2: re-enable via hysteresis (need 2 consecutive "good" updates) ---
    # Choose lp_i large enough to exceed hysteresis_up_mult * τ_LP
    for _ in range(2):
        g = mod._maybe_update_gate(0, lp_i=1.0)
        assert g in (0, 1)
    assert mod._stats[0].gate == 1, "Region should re-enable after hysteresis criterion is met"
