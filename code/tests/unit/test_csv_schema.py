from __future__ import annotations

from pathlib import Path

from irl.utils.loggers import CSVLogger


def test_csvlogger_does_not_crash_on_new_keys(tmp_path: Path) -> None:
    p = tmp_path / "scalars.csv"
    log = CSVLogger(p)
    try:
        log.log_row(0, {"a": 1.0})
        log.log_row(1, {"a": 2.0, "b": 3.0})
    finally:
        log.close()

    text = p.read_text(encoding="utf-8").strip().splitlines()
    assert text
    assert text[0].startswith("step,")
    assert "b" not in text[0]
