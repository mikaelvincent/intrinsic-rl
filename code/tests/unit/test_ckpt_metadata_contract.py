from __future__ import annotations

import pytest

from irl.experiments.evaluation import _cfg_fields as cfg_fields_suite
from irl.pipelines.eval import _cfg_fields as cfg_fields_pipeline


@pytest.mark.parametrize(
    "payload, expected",
    [
        (
            {"cfg": {"env": {"id": "DummyEval-v0"}, "method": "vanilla", "seed": 7}},
            ("DummyEval-v0", "vanilla", 7),
        ),
        (
            {"cfg": {"env": {"id": 123}, "method": None, "seed": "42"}},
            ("123", None, 42),
        ),
        (
            {"cfg": {"env": {"id": "EnvA"}, "method": "glpe", "seed": "notint"}},
            ("EnvA", "glpe", None),
        ),
        (
            {"cfg": "not-a-mapping"},
            (None, None, None),
        ),
        (
            {},
            (None, None, None),
        ),
        (
            {"cfg": {"env": {}, "method": "vanilla", "seed": 1}},
            (None, "vanilla", 1),
        ),
    ],
)
def test_cfg_fields_extraction_matches_across_modules(payload: dict, expected: tuple) -> None:
    assert cfg_fields_pipeline(payload) == expected
    assert cfg_fields_suite(payload) == expected
