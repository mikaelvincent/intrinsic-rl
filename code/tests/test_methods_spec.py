from __future__ import annotations

from typing import get_args

from irl.cfg.schema import MethodLiteral
from irl.intrinsic.config import build_intrinsic_kwargs
from irl.intrinsic.factory import _canonical_method
from irl.trainer.update_steps import _canonical_intrinsic_method


def test_method_normalization_and_glpe_cache_rules() -> None:
    methods = [str(m) for m in get_args(MethodLiteral)]
    for m in methods:
        ml = str(m).lower()
        expected = "glpe" if ml.startswith("glpe_") else ml
        assert _canonical_method(m) == expected
        assert _canonical_intrinsic_method(m) == expected

    out_nogate = build_intrinsic_kwargs({"method": "glpe_nogate", "intrinsic": {"gate": {"enabled": True}}})
    assert out_nogate["gating_enabled"] is False

    cache_interval = 64
    out_cache = build_intrinsic_kwargs(
        {"method": "glpe_cache", "intrinsic": {"gate": {"median_cache_interval": cache_interval}}}
    )
    assert int(out_cache["gate_median_cache_interval"]) == cache_interval

    for m in methods:
        ml = str(m).lower()
        if not ml.startswith("glpe") or ml == "glpe_cache":
            continue
        out = build_intrinsic_kwargs(
            {"method": ml, "intrinsic": {"gate": {"median_cache_interval": cache_interval}}}
        )
        assert int(out["gate_median_cache_interval"]) == 1
