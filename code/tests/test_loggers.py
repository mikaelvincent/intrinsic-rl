from __future__ import annotations

import csv
from pathlib import Path

from irl.cfg.schema import LoggingConfig
from irl.utils.loggers import MetricLogger


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_metric_logger_writes_nonfinite_values(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ml = MetricLogger(run_dir, LoggingConfig(csv_interval=1, checkpoint_interval=100_000))
    try:
        assert ml.log(step=0, foo=float("nan"), bar=1.0) is True
        assert ml.log(step=1, foo=0.0, bar=float("inf")) is True
    finally:
        ml.close()

    rows = _read_rows(run_dir / "logs" / "scalars.csv")
    assert len(rows) >= 2

    r0 = rows[0]
    assert r0["foo"].strip().lower() == "nan"
    assert int(float(r0["nonfinite_any"])) == 1
    assert int(float(r0["nonfinite_count"])) == 1
    assert "foo" in {k for k in r0["nonfinite_keys"].split(",") if k}

    r1 = rows[1]
    assert r1["bar"].strip().lower() == "inf"
    assert int(float(r1["nonfinite_any"])) == 1
    assert int(float(r1["nonfinite_count"])) == 1
    assert "bar" in {k for k in r1["nonfinite_keys"].split(",") if k}
