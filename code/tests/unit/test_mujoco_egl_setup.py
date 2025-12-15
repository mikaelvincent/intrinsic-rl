from __future__ import annotations

import os
import sys
from ctypes import CDLL
from ctypes.util import find_library

import pytest

from irl.trainer.build import ensure_mujoco_gl


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="EGL setup check is Linux-only")
def test_system_egl_loadable(monkeypatch: pytest.MonkeyPatch) -> None:
    lib_path = find_library("EGL")
    if not lib_path:
        pytest.skip("EGL library not installed on this system")

    try:
        CDLL(lib_path)
    except OSError as exc:
        pytest.fail(f"Found EGL library at {lib_path} but failed to load it: {exc}")

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    backend = ensure_mujoco_gl("Ant-v4")

    assert backend == "egl"
    assert os.environ.get("MUJOCO_GL") == "egl"
