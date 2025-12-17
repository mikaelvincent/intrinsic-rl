from __future__ import annotations

from pathlib import Path

import pytest

from irl.utils.io import atomic_write_csv


def test_atomic_write_csv_overwrites_and_cleans_tmp(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    tmp = path.with_suffix(path.suffix + ".tmp")

    atomic_write_csv(path, ["a", "b"], [{"a": 1, "b": "x"}])

    assert path.exists()
    assert not tmp.exists()
    assert path.read_text(encoding="utf-8").splitlines() == ["a,b", "1,x"]

    atomic_write_csv(path, ["a", "b"], [{"a": 2, "b": "y"}])

    assert path.read_text(encoding="utf-8").splitlines() == ["a,b", "2,y"]
    assert not tmp.exists()


def test_atomic_write_csv_rejects_extra_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    with pytest.raises(ValueError):
        atomic_write_csv(path, ["a"], [{"a": 1, "b": 2}])
