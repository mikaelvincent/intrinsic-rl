from __future__ import annotations

import inspect
from pathlib import Path

from typer.models import OptionInfo

import irl.eval as eval_cmd
import irl.experiments as suite_cmd
import irl.train as train_cmd


def _opt_default(fn: object, name: str) -> object:
    sig = inspect.signature(fn)
    p = sig.parameters[name]
    assert isinstance(p.default, OptionInfo)
    return p.default.default


def test_train_cli_defaults() -> None:
    assert _opt_default(train_cmd.cli_train, "config") is None
    assert _opt_default(train_cmd.cli_train, "total_steps") is None
    assert _opt_default(train_cmd.cli_train, "run_dir") is None
    assert _opt_default(train_cmd.cli_train, "method") is None
    assert _opt_default(train_cmd.cli_train, "env") is None
    assert _opt_default(train_cmd.cli_train, "device") is None
    assert _opt_default(train_cmd.cli_train, "resume") is False


def test_suite_train_cli_defaults() -> None:
    assert _opt_default(suite_cmd.cli_train, "configs_dir") == Path("configs")
    assert _opt_default(suite_cmd.cli_train, "total_steps") == 150_000
    assert _opt_default(suite_cmd.cli_train, "runs_root") == Path("runs_suite")
    assert _opt_default(suite_cmd.cli_train, "resume") is True
    assert _opt_default(suite_cmd.cli_train, "auto_async") is False


def test_eval_cli_defaults() -> None:
    assert _opt_default(eval_cmd.cli_eval, "episodes") == 20
    assert _opt_default(eval_cmd.cli_eval, "device") == "cpu"
    assert _opt_default(eval_cmd.cli_eval, "policy") == "mode"
    assert _opt_default(eval_cmd.cli_eval, "quick") is False
    assert _opt_default(eval_cmd.cli_eval, "out") is None


def test_suite_eval_cli_defaults() -> None:
    assert _opt_default(suite_cmd.cli_eval, "runs_root") == Path("runs_suite")
    assert _opt_default(suite_cmd.cli_eval, "results_dir") == Path("results_suite")
    assert _opt_default(suite_cmd.cli_eval, "episodes") == 20
    assert _opt_default(suite_cmd.cli_eval, "device") == "cpu"
    assert _opt_default(suite_cmd.cli_eval, "policy") == "mode"
    assert _opt_default(suite_cmd.cli_eval, "quick") is False
    assert _opt_default(suite_cmd.cli_eval, "strict_coverage") is True
    assert _opt_default(suite_cmd.cli_eval, "strict_step_parity") is True
