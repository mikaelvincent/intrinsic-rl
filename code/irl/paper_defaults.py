from __future__ import annotations

from pathlib import Path
from typing import Final

CONFIGS_DIR: Final[Path] = Path("configs")
RUNS_ROOT: Final[Path] = Path("runs_suite")
RESULTS_DIR: Final[Path] = Path("results_suite")
BENCH_DIR: Final[Path] = Path("results/benchmarks")

DEFAULT_TRAIN_TOTAL_STEPS: Final[int] = 150_000
DEFAULT_EVAL_EPISODES: Final[int] = 20
DEFAULT_EVAL_POLICY_MODE: Final[str] = "mode"

DEFAULT_VIDEO_POLICY_MODE: Final[str] = "mode"
DEFAULT_VIDEO_SEEDS: Final[tuple[int, ...]] = (100,)
DEFAULT_VIDEO_MAX_STEPS: Final[int] = 1000
DEFAULT_VIDEO_FPS: Final[int] = 30

DEFAULT_BENCH_DEVICE: Final[str] = "cpu"
DEFAULT_BENCH_THREADS: Final[int] = 1
DEFAULT_BENCH_SEED: Final[int] = 0
DEFAULT_BENCH_QUICK: Final[bool] = True
