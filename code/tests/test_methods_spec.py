from __future__ import annotations

from typing import get_args

from irl.cfg.schema import MethodLiteral
from irl.intrinsic.config import build_intrinsic_kwargs
from irl.intrinsic.factory import _canonical_method
from irl.trainer.update_steps import _canonical_intrinsic_method
from irl.visualization.paper_figures import paper_method_groups


def test_glpe_variants_canonicalize_to_glpe() -> None:
    methods = [str(m) for m in get_args(MethodLiteral)]
    for m in methods:
        ml = str(m).lower()
        expected = "glpe" if ml.startswith("glpe_") else ml
        assert _canonical_method(m) == expected
        assert _canonical_intrinsic_method(m) == expected


def test_build_intrinsic_kwargs_enforces_glpe_gate_and_cache_rules() -> None:
    cfg_nogate = {"method": "glpe_nogate", "intrinsic": {"gate": {"enabled": True}}}
    out_nogate = build_intrinsic_kwargs(cfg_nogate)
    assert out_nogate["gating_enabled"] is False

    cache_interval = 64
    cfg_cache = {
        "method": "glpe_cache",
        "intrinsic": {"gate": {"median_cache_interval": cache_interval}},
    }
    out_cache = build_intrinsic_kwargs(cfg_cache)
    assert int(out_cache["gate_median_cache_interval"]) == cache_interval

    for m in get_args(MethodLiteral):
        ml = str(m).lower()
        if not ml.startswith("glpe") or ml == "glpe_cache":
            continue
        cfg = {
            "method": ml,
            "intrinsic": {"gate": {"median_cache_interval": cache_interval}},
        }
        out = build_intrinsic_kwargs(cfg)
        assert int(out["gate_median_cache_interval"]) == 1


def test_paper_method_groups_order_is_stable() -> None:
    methods = [
        "GLPE",
        "vanilla",
        "Foo",
        "glpe_cache",
        "RIAC",
        "glpe_nogate",
        "bar",
        "glpe_lp_only",
        "glpe_zeta",
        "ride",
        "icm",
        "rnd",
    ]
    baselines, ablations = paper_method_groups(methods)

    assert baselines == ["vanilla", "icm", "rnd", "ride", "riac", "glpe", "bar", "foo"]
    assert ablations == ["glpe_lp_only", "glpe_nogate", "glpe_cache", "glpe", "glpe_zeta"]
