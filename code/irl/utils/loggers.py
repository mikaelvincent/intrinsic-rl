from __future__ import annotations

import csv
import logging
from logging import Logger
from pathlib import Path
from typing import Mapping, Optional

from irl.cfg.schema import LoggingConfig

_DEFAULT_LOGGER_NAME = "irl"


def get_logger(name: str | None = None) -> Logger:
    base = logging.getLogger(_DEFAULT_LOGGER_NAME)
    if not base.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        base.addHandler(handler)
        base.setLevel(logging.INFO)
        base.propagate = False
    return base if name is None else base.getChild(str(name))


def log_resume_loaded(step: int, ckpt_path: Path | str) -> None:
    get_logger("resume").info("Loaded latest checkpoint at step=%d from %s", step, ckpt_path)


def log_resume_no_checkpoint() -> None:
    get_logger("resume").info("No checkpoint found; starting a new run.")


def log_resume_state_restored(global_step: int) -> None:
    get_logger("resume").info("State restored. Continuing from global_step=%d.", global_step)


def log_resume_optimizer_warning() -> None:
    get_logger("resume").warning("PPO optimizer state not restored; continuing with fresh optimizers.")


def log_resume_intrinsic_warning(method: str | None = None) -> None:
    msg = (
        f"Intrinsic module state not restored for method={method!r}."
        if method
        else "Intrinsic module state not restored."
    )
    get_logger("resume").warning(msg)


def log_mujoco_gl_preserve(current: str) -> None:
    get_logger("mujoco").info("MUJOCO_GL=%s (pre-set).", current)


def log_mujoco_gl_default(value: str) -> None:
    get_logger("mujoco").info(
        "MUJOCO_GL not set; defaulting to %r for headless MuJoCo rendering.", value
    )


def log_intrinsic_norm_hint(method: str, outputs_normalized: bool) -> None:
    log = get_logger("intrinsic")
    m = str(method).lower()

    if outputs_normalized:
        if m == "glpe":
            log.info(
                "Intrinsic normalization: method=%r normalizes impact+LP inside the module "
                "(normalize_inside=True, outputs_normalized=True); trainer's global "
                "RunningRMS for intrinsic is disabled.",
                method,
            )
        elif m == "riac":
            log.info(
                "Intrinsic normalization: method=%r normalizes learning-progress inside "
                "the module (internal RunningRMS over LP); trainer's global RunningRMS "
                "for intrinsic is disabled.",
                method,
            )
        elif m == "rnd":
            log.info(
                "Intrinsic normalization: method=%r uses RNDConfig.normalize_intrinsic=True "
                "(module-owned RMS over prediction error); trainer's global RunningRMS "
                "for intrinsic is disabled.",
                method,
            )
        else:
            log.info(
                "Intrinsic normalization: method=%r outputs are normalized inside the "
                "module (module-owned RMS, outputs_normalized=True); trainer will NOT "
                "apply its global RunningRMS to intrinsic rewards.",
                method,
            )
    else:
        log.info(
            "Intrinsic normalization: method=%r outputs are raw; trainer applies a global "
            "RunningRMS over intrinsic rewards before clipping and scaling.",
            method,
        )


def log_domain_randomization(summary: str) -> None:
    get_logger("env").info("Domain randomization applied on env.reset(): %s", summary)


class CSVLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[list[str]] = None
        self._wrote_header = self.path.exists() and self.path.stat().st_size > 0

    def _ensure_writer(self, row: Mapping[str, object]) -> None:
        if self._writer is not None:
            return
        keys = [k for k in row.keys() if k != "step"]
        self._fieldnames = ["step"] + sorted(keys)
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        if not self._wrote_header:
            self._writer.writeheader()
            self._file.flush()
            self._wrote_header = True

    def log_row(self, step: int, metrics: Mapping[str, object]) -> None:
        row = {"step": int(step)}
        for k, v in metrics.items():
            if k == "step":
                continue
            if isinstance(v, (int, float, str, bool)):
                row[k] = v
            else:
                row[k] = str(v)
        self._ensure_writer(row)
        assert self._writer is not None
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.flush()
        finally:
            self._file.close()


class MetricLogger:
    def __init__(self, run_dir: Path, cfg: LoggingConfig) -> None:
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.run_dir / "logs" / "scalars.csv"
        self.csv = CSVLogger(self.csv_path)

        self.tb = None
        self._last_csv_write_step: Optional[int] = None

    def log(self, step: int, **metrics: float) -> None:
        interval = int(max(1, self.cfg.csv_interval))
        s = int(step)
        last = self._last_csv_write_step

        should_write_csv = False
        if last is None:
            if s == 0 or s >= interval:
                should_write_csv = True
        else:
            if s >= last + interval:
                should_write_csv = True

        if should_write_csv:
            self.csv.log_row(s, metrics)
            self._last_csv_write_step = s

    def log_hparams(self, params: Mapping[str, object]) -> None:
        return

    def close(self) -> None:
        self.csv.close()
