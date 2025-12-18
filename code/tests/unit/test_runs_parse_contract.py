from __future__ import annotations

import pytest

from irl.utils.runs import parse_run_name


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "glpe__MountainCar-v0__seed42__cfgA",
            {"method": "glpe", "env": "MountainCar-v0", "seed": "42"},
        ),
        (
            "vanilla__CartPole-v1__seed1__cfgA__extra",
            {"method": "vanilla", "env": "CartPole-v1", "seed": "1"},
        ),
        ("vanilla", {"method": "vanilla"}),
        ("vanilla__CartPole-v1", {"method": "vanilla", "env": "CartPole-v1"}),
        ("vanilla__DummyEval-v0__notseed5__cfg", {"method": "vanilla", "env": "DummyEval-v0"}),
        (
            "vanilla__DummyEval-v0__seed007__20250101-000000",
            {"method": "vanilla", "env": "DummyEval-v0", "seed": "007"},
        ),
    ],
)
def test_parse_run_name_contract(name: str, expected: dict[str, str]) -> None:
    assert parse_run_name(name) == expected
