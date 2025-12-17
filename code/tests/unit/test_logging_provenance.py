from __future__ import annotations

import json
from pathlib import Path

from irl.cfg.schema import LoggingConfig
from irl.utils.checkpoint import compute_cfg_hash
from irl.utils.loggers import CSVLogger, MetricLogger


def test_log_hparams_writes_config_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ml = MetricLogger(run_dir, LoggingConfig(csv_interval=1, checkpoint_interval=10_000))

    try:
        params = {
            "seed": 7,
            "device": "cpu",
            "env": {"id": "MountainCar-v0", "vec_envs": 2},
            "exp": {"deterministic": True},
        }

        ml.log_hparams(params)

        cfg_path = run_dir / "config.json"
        hash_path = run_dir / "config_hash.txt"

        assert cfg_path.exists()
        assert hash_path.exists()

        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)

        expected_hash = compute_cfg_hash(params)
        assert payload.get("cfg_hash") == expected_hash
        assert hash_path.read_text(encoding="utf-8").strip() == expected_hash

        before = cfg_path.read_text(encoding="utf-8")
        ml.log_hparams({"seed": 999})
        after = cfg_path.read_text(encoding="utf-8")

        assert after == before
    finally:
        ml.close()

    p = tmp_path / "scalars_extra.csv"
    log = CSVLogger(p)
    try:
        log.log_row(0, {"a": 1.0})
        log.log_row(1, {"a": 2.0, "b": 3.0})
    finally:
        log.close()

    header = p.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "b" not in {c.strip() for c in header}
