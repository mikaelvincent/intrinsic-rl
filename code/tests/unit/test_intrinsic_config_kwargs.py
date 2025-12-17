from __future__ import annotations

from dataclasses import replace

from irl.cfg import Config, to_dict
from irl.intrinsic.config import build_intrinsic_kwargs


def test_build_intrinsic_kwargs_matches_dataclass_and_dict() -> None:
    base = Config()
    cfg = replace(
        base,
        method="glpe",
        intrinsic=replace(
            base.intrinsic,
            eta=0.1,
            alpha_impact=1.25,
            alpha_lp=0.75,
            bin_size=0.33,
            region_capacity=321,
            depth_max=5,
            ema_beta_long=0.987,
            ema_beta_short=0.876,
            normalize_inside=False,
            checkpoint_include_points=False,
            gate=replace(
                base.intrinsic.gate,
                enabled=False,
                tau_lp_mult=0.123,
                tau_s=1.5,
                hysteresis_up_mult=1.7,
                min_consec_to_gate=4,
                min_regions_for_gating=2,
            ),
        ),
    )

    kw_dc = build_intrinsic_kwargs(cfg)
    kw_dict = build_intrinsic_kwargs(to_dict(cfg))
    assert kw_dc == kw_dict


def test_build_intrinsic_kwargs_glpe_nogate_forces_disabled() -> None:
    kw = build_intrinsic_kwargs({"method": "glpe_nogate", "intrinsic": {"gate": {"enabled": True}}})
    assert kw["gating_enabled"] is False
