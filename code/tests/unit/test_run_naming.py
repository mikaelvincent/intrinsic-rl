from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from irl.cfg import Config
from irl.experiments.training import _run_dir_for
from irl.utils.runs import parse_run_name


def test_run_dir_for_parses_roundtrip(tmp_path: Path) -> None:
    base = Config()
    cfg = replace(base, method="glpe", env=replace(base.env, id="Foo/Bar-v0"), seed=7)

    cfg_path = tmp_path / "cfgA.yaml"
    cfg_path.write_text("seed: 7\n", encoding="utf-8")

    run_dir = _run_dir_for(cfg, cfg_path, seed=int(cfg.seed), runs_root=tmp_path)
    info = parse_run_name(run_dir.name)

    assert info.get("method") == "glpe"
    assert info.get("env") == "Foo-Bar-v0"
    assert info.get("seed") == "7"
