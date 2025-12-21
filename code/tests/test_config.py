from __future__ import annotations

import csv
from pathlib import Path

import pytest

from irl.cfg import ConfigError, loads_config
from irl.cfg.schema import LoggingConfig
from irl.intrinsic.config import build_intrinsic_kwargs
from irl.utils.config_hash import compute_cfg_hash
from irl.utils.loggers import MetricLogger


def test_loads_config_parses_numbers_and_validates() -> None:
    cfg = loads_config(
        """
method: vanilla
env:
  vec_envs: "8"
ppo:
  steps_per_update: "2_048"
  minibatches: 3_2
  learning_rate: "3e-4"
""".lstrip()
    )
    assert int(cfg.env.vec_envs) == 8
    assert int(cfg.ppo.steps_per_update) == 2048
    assert int(cfg.ppo.minibatches) == 32
    assert abs(float(cfg.ppo.learning_rate) - 3e-4) < 1e-12

    with pytest.raises(ConfigError):
        loads_config("seed: true\n")

    with pytest.raises(ConfigError):
        loads_config("seed: 1\nunknown_top_level: 123\n")

    with pytest.raises(ConfigError):
        loads_config(
            """
method: vanilla
env:
  vec_envs: 8
ppo:
  steps_per_update: 130
  minibatches: 64
""".lstrip()
        )


def test_taper_fracs_validate_for_glpe_only() -> None:
    cfg = loads_config(
        """
method: glpe
intrinsic:
  taper_start_frac: 0.1
  taper_end_frac: 0.9
""".lstrip()
    )
    assert cfg.intrinsic.taper_start_frac == pytest.approx(0.1)
    assert cfg.intrinsic.taper_end_frac == pytest.approx(0.9)

    with pytest.raises(
        ConfigError,
        match=r"Set both `intrinsic\.taper_start_frac` and `intrinsic\.taper_end_frac`, or omit both\.",
    ):
        loads_config(
            """
method: glpe
intrinsic:
  taper_start_frac: 0.1
""".lstrip()
        )

    for bad_start, bad_end in ((-0.1, 0.5), (0.1, 1.1), (0.5, 0.5), (0.7, 0.6)):
        with pytest.raises(ConfigError, match=r"0\.0 <= start < end <= 1\.0"):
            loads_config(
                f"""
method: glpe
intrinsic:
  taper_start_frac: {bad_start}
  taper_end_frac: {bad_end}
""".lstrip()
            )

    with pytest.raises(ConfigError, match=r"glpe\* methods"):
        loads_config(
            """
method: vanilla
intrinsic:
  taper_start_frac: 0.1
  taper_end_frac: 0.9
""".lstrip()
        )


def test_compute_cfg_hash_ignores_none_taper_keys() -> None:
    a = {"intrinsic": {}}
    b = {"intrinsic": {"taper_start_frac": None, "taper_end_frac": None}}
    assert compute_cfg_hash(a) == compute_cfg_hash(b)


def test_build_intrinsic_kwargs_enforces_glpe_gate_cache_rules() -> None:
    out_nogate = build_intrinsic_kwargs(
        {"method": "glpe_nogate", "intrinsic": {"gate": {"enabled": True}}}
    )
    assert out_nogate["gating_enabled"] is False

    cache_interval = 64
    out_cache = build_intrinsic_kwargs(
        {"method": "glpe_cache", "intrinsic": {"gate": {"median_cache_interval": cache_interval}}}
    )
    assert int(out_cache["gate_median_cache_interval"]) == cache_interval

    for m in ("glpe", "glpe_lp_only", "glpe_impact_only", "glpe_nogate"):
        out = build_intrinsic_kwargs(
            {"method": m, "intrinsic": {"gate": {"median_cache_interval": cache_interval}}}
        )
        assert int(out["gate_median_cache_interval"]) == 1


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_metric_logger_handles_nonfinite_and_schema_expansion(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ml = MetricLogger(run_dir, LoggingConfig(csv_interval=1, checkpoint_interval=100_000))
    try:
        assert ml.log(step=0, foo=float("nan"), bar=1.0) is True
        assert ml.log(step=1, foo=0.0, bar=float("inf")) is True
    finally:
        ml.close()

    rows = _read_rows(run_dir / "logs" / "scalars.csv")
    assert len(rows) >= 2

    r0 = rows[0]
    assert r0["foo"].strip().lower() == "nan"
    assert int(float(r0["nonfinite_any"])) == 1
    assert "foo" in {k for k in r0["nonfinite_keys"].split(",") if k}

    r1 = rows[1]
    assert r1["bar"].strip().lower() == "inf"
    assert int(float(r1["nonfinite_any"])) == 1
    assert "bar" in {k for k in r1["nonfinite_keys"].split(",") if k}

    assert "bar" in r0
    assert (r0["bar"] or "").strip() != ""
