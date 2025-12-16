from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _RegionStats:
    ema_long: float = 0.0
    ema_short: float = 0.0
    count: int = 0
    gate: int = 1
    bad_consec: int = 0
    good_consec: int = 0


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
    if not sufficient_regions:
        st.bad_consec = 0
        st.good_consec = 0
        st.gate = 1
        return st.gate

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
        if lp_i > (hysteresis_up_mult * tau_lp):
            st.good_consec += 1
            if st.good_consec >= 2:
                st.gate = 1
                st.bad_consec = 0
        else:
            st.good_consec = 0

    return st.gate
