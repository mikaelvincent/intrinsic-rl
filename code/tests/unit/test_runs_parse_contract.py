from __future__ import annotations

from irl.utils.runs import parse_run_name


def test_parse_run_name_standard_pattern() -> None:
    name = "glpe__MountainCar-v0__seed42__cfgA"
    assert parse_run_name(name) == {"method": "glpe", "env": "MountainCar-v0", "seed": "42"}


def test_parse_run_name_ignores_extra_tokens() -> None:
    name = "vanilla__CartPole-v1__seed1__cfgA__extra"
    assert parse_run_name(name) == {"method": "vanilla", "env": "CartPole-v1", "seed": "1"}


def test_parse_run_name_missing_parts() -> None:
    assert parse_run_name("vanilla") == {"method": "vanilla"}
    assert parse_run_name("vanilla__CartPole-v1") == {"method": "vanilla", "env": "CartPole-v1"}


def test_parse_run_name_seed_must_be_third_token() -> None:
    name = "vanilla__DummyEval-v0__notseed5__cfg"
    assert parse_run_name(name) == {"method": "vanilla", "env": "DummyEval-v0"}


def test_parse_run_name_preserves_seed_string() -> None:
    name = "vanilla__DummyEval-v0__seed007__20250101-000000"
    assert parse_run_name(name) == {"method": "vanilla", "env": "DummyEval-v0", "seed": "007"}
