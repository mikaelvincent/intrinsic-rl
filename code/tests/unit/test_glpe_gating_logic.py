from irl.intrinsic.glpe.gating import _RegionStats, update_region_gate


def test_update_region_gate_gates_then_recovers():
    st = _RegionStats(ema_long=10.0, ema_short=10.0, count=10, gate=1)

    for _ in range(2):
        g = update_region_gate(
            st,
            lp_i=0.0,
            tau_lp=1.0,
            tau_s=0.5,
            median_error_global=1.0,
            hysteresis_up_mult=1.1,
            min_consec_to_gate=3,
            sufficient_regions=True,
        )
        assert g == 1

    g = update_region_gate(
        st,
        lp_i=0.0,
        tau_lp=1.0,
        tau_s=0.5,
        median_error_global=1.0,
        hysteresis_up_mult=1.1,
        min_consec_to_gate=3,
        sufficient_regions=True,
    )
    assert g == 0
    assert st.gate == 0

    g = update_region_gate(
        st,
        lp_i=1.2,
        tau_lp=1.0,
        tau_s=0.5,
        median_error_global=1.0,
        hysteresis_up_mult=1.1,
        min_consec_to_gate=3,
        sufficient_regions=True,
    )
    assert g == 0

    g = update_region_gate(
        st,
        lp_i=1.2,
        tau_lp=1.0,
        tau_s=0.5,
        median_error_global=1.0,
        hysteresis_up_mult=1.1,
        min_consec_to_gate=3,
        sufficient_regions=True,
    )
    assert g == 1
    assert st.gate == 1


def test_update_region_gate_resets_when_insufficient():
    st = _RegionStats(ema_long=1.0, ema_short=10.0, count=10, gate=0, bad_consec=5, good_consec=1)

    g = update_region_gate(
        st,
        lp_i=0.0,
        tau_lp=1.0,
        tau_s=2.0,
        median_error_global=1.0,
        hysteresis_up_mult=2.0,
        min_consec_to_gate=3,
        sufficient_regions=False,
    )
    assert g == 1
    assert st.gate == 1
    assert st.bad_consec == 0
    assert st.good_consec == 0
