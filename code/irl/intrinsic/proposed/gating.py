"""Region gating helpers for the Proposed intrinsic module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _RegionStats:
    """Per-region EMA container + gating state."""

    ema_long: float = 0.0
    ema_short: float = 0.0
    count: int = 0
    # gating state
    gate: int = 1  # 1 => enabled, 0 => gated-off
    bad_consec: int = 0  # consecutive 'bad' refreshes toward gating off
    good_consec: int = 0  # consecutive 'good' refreshes toward re-enable


def update_region_gate(
    st: _RegionStats,
    *,
    lp_i: float,
    tau_lp: float,
    tau_s: float,
    median_error_global: float,
    hysteresis_up_mult: float,
    min_consec_to_gate: int,
    eps: float = 1e-8,
    sufficient_regions: bool = True,
) -> int:
    """Update and return region gate (1 enabled, 0 gated-off).

    Args:
        st: region stats object (updated in-place).
        lp_i: current LP value for this region.
        tau_lp: LP threshold (absolute).
        tau_s: stochasticity threshold.
        median_error_global: median EMA_short across regions (for S_i).
        hysteresis_up_mult: multiplier for τ_LP to re-enable a gated region.
        min_consec_to_gate: consecutive 'bad' refreshes required to gate off.
        eps: numerical guard.
        sufficient_regions: if False, keep gate open and reset counters.

    Returns:
        int: current gate value (1 or 0).
    """
    if not sufficient_regions:
        st.bad_consec = 0
        st.good_consec = 0
        st.gate = 1
        return st.gate

    # Stochasticity score S_i = EMA_short / (eps + median_error_global)
    s_i = float(st.ema_short) / (eps + float(median_error_global))

    cond_bad = (lp_i < tau_lp) and (s_i > tau_s)

    if st.gate == 1:
        if cond_bad:
            st.bad_consec += 1
            if st.bad_consec >= int(min_consec_to_gate):
                st.gate = 0
                st.good_consec = 0
        else:
            st.bad_consec = 0
    else:
        # hysteresis to re-enable
        if lp_i > (hysteresis_up_mult * tau_lp):
            st.good_consec += 1
            if st.good_consec >= 2:
                st.gate = 1
                st.bad_consec = 0
        else:
            st.good_consec = 0

    return st.gate
