from __future__ import annotations

from dataclasses import replace

from irl.cfg.schema import Config
from irl.utils.steps import resolve_total_steps


def test_resolve_total_steps_clamps_below_vec_envs_for_override() -> None:
    base = Config()
    cfg = replace(base, env=replace(base.env, vec_envs=8))

    out = resolve_total_steps(cfg, requested_steps=5, align_to_vec_envs=True)
    assert int(out) == 8


def test_resolve_total_steps_clamps_below_vec_envs_for_cfg_total_steps() -> None:
    base = Config()
    cfg = replace(
        base,
        env=replace(base.env, vec_envs=8),
        exp=replace(base.exp, total_steps=5),
    )

    out = resolve_total_steps(cfg, requested_steps=None, align_to_vec_envs=True)
    assert int(out) == 8


def test_resolve_total_steps_floors_to_multiple_when_possible() -> None:
    base = Config()
    cfg = replace(base, env=replace(base.env, vec_envs=8))

    out = resolve_total_steps(cfg, requested_steps=17, align_to_vec_envs=True)
    assert int(out) == 16


def test_resolve_total_steps_no_align_for_single_env() -> None:
    base = Config()
    cfg = replace(base, env=replace(base.env, vec_envs=1))

    out = resolve_total_steps(cfg, requested_steps=5, align_to_vec_envs=True)
    assert int(out) == 5
