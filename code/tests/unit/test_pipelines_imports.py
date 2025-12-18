from __future__ import annotations

import importlib


def test_pipelines_package_exports_and_imports() -> None:
    pkg = importlib.import_module("irl.pipelines")

    for name in (
        "discover_run_dirs_with_latest_ckpt",
        "discover_run_dirs_with_step_ckpts",
        "collect_ckpts_from_patterns",
    ):
        assert hasattr(pkg, name)

    importlib.import_module("irl.pipelines.discovery")
    importlib.import_module("irl.pipelines.eval")
    importlib.import_module("irl.pipelines.policy_rollout")
    importlib.import_module("irl.pipelines.runtime")
