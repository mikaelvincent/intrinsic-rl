from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


_SCRIPT = r"""
import json
from irl.visualization.paper.eval_plots import _color_for_method
from irl.visualization.paper.glpe_plots import _sample_idx, _sample_seed

methods = [
    "icm",
    "rnd",
    "ride",
    "riac",
    "foo",
    "bar",
    "baz",
    "qux",
    "quux",
    "corge",
    "grault",
    "garply",
]

out = {
    "colors": [_color_for_method(m) for m in methods],
    "seed_gate": int(_sample_seed("glpe_gate_map", "MountainCar-v0")),
    "seed_extint": int(_sample_seed("glpe_extint", "MountainCar-v0")),
    "idx_gate": _sample_idx(100, 10, seed=_sample_seed("glpe_gate_map", "MountainCar-v0")).tolist(),
    "idx_extint": _sample_idx(100, 10, seed=_sample_seed("glpe_extint", "MountainCar-v0")).tolist(),
}

print(json.dumps(out, sort_keys=True))
"""


def _run_with_hashseed(hash_seed: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hash_seed)

    py_path = str(root)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = py_path if not existing else (py_path + os.pathsep + existing)

    out = subprocess.check_output(
        [sys.executable, "-c", _SCRIPT],
        env=env,
        cwd=str(root),
        text=True,
    ).strip()
    return json.loads(out)


def test_plots_deterministic_across_pythonhashseed() -> None:
    a = _run_with_hashseed("1")
    b = _run_with_hashseed("2")
    assert a == b
