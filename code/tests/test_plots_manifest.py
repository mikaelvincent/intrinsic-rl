from __future__ import annotations

import json
from pathlib import Path

from irl.experiments.plotting import run_plots_suite


def _write_min_summary_csv(path: Path) -> None:
    path.write_text(
        (
            "method,env_id,mean_return_mean,mean_return_ci95_lo,mean_return_ci95_hi,n_seeds\n"
            "vanilla,DummyEval-v0,1.0,0.8,1.2,1\n"
        ),
        encoding="utf-8",
    )


def test_plots_manifest_created_and_nonempty(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs_root"
    results_dir = tmp_path / "results_dir"
    runs_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    _write_min_summary_csv(results_dir / "summary.csv")

    run_plots_suite(runs_root=runs_root, results_dir=results_dir)

    manifest = results_dir / "plots" / "plots_manifest.json"
    assert manifest.exists()

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert isinstance(data, dict)

    assert int(data.get("n_written", 0)) > 0
    written = data.get("written")
    assert isinstance(written, list)
    assert written

    p0 = Path(str(written[0]))
    if not p0.is_absolute():
        p0 = results_dir / p0
    assert p0.exists()
