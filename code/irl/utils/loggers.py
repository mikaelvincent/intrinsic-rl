"""CSV/TensorBoard scalar logging with simple cadence.

CSV writes on a configured interval; TensorBoard (if available) logs each call.

This module also exposes a lightweight console logger for human-readable
informational messages emitted by training utilities (resume notices,
MuJoCo hints, intrinsic-normalization hints, etc.).
"""

from __future__ import annotations

import csv
import logging
from logging import Logger
from pathlib import Path
from typing import Mapping, Optional

from irl.cfg.schema import LoggingConfig

# TensorBoard is an optional runtime path if disabled via config;
# import lazily to avoid overhead when not used.
try:  # pragma: no cover - import guard
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - safe fallback
    SummaryWriter = None  # type: ignore


# ---------------------------------------------------------------------------
# Lightweight console logger
# ---------------------------------------------------------------------------

_DEFAULT_LOGGER_NAME = "irl"


def get_logger(name: str | None = None) -> Logger:
    """Return a library logger for human-readable console output.

    The first call installs a default ``StreamHandler`` if the ``irl`` logger
    has no handlers yet, so applications may override configuration by
    attaching their own handlers beforehand.
    """
    base = logging.getLogger(_DEFAULT_LOGGER_NAME)
    if not base.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        base.addHandler(handler)
        base.setLevel(logging.INFO)
        # Avoid double logging through the root logger unless the user
        # explicitly wires it up.
        base.propagate = False
    return base if name is None else base.getChild(str(name))


def log_resume_loaded(step: int, ckpt_path: Path | str) -> None:
    """Standardized message when a checkpoint is successfully loaded."""
    get_logger("resume").info(
        "Loaded latest checkpoint at step=%d from %s", step, ckpt_path
    )


def log_resume_no_checkpoint() -> None:
    """Standardized message when no checkpoint is found for resume."""
    get_logger("resume").info(
        "No checkpoint found; starting a new run."
    )


def log_resume_state_restored(global_step: int) -> None:
    """Standardized message after state is fully restored."""
    get_logger("resume").info(
        "State restored. Continuing from global_step=%d.", global_step
    )


def log_resume_optimizer_warning() -> None:
    """Warn when PPO optimizer state cannot be restored."""
    get_logger("resume").warning(
        "PPO optimizer state not restored; continuing with fresh optimizers."
    )


def log_resume_intrinsic_warning(method: str | None = None) -> None:
    """Warn when intrinsic module state cannot be restored."""
    if method:
        msg = f"Intrinsic module state not restored for method={method!r}."
    else:
        msg = "Intrinsic module state not restored."
    get_logger("resume").warning(msg)


def log_mujoco_gl_preserve(current: str) -> None:
    """Informational message when an existing MUJOCO_GL is preserved."""
    get_logger("mujoco").info("MUJOCO_GL=%s (pre-set).", current)


def log_mujoco_gl_default(value: str) -> None:
    """Informational message when MUJOCO_GL is defaulted for headless use."""
    get_logger("mujoco").info(
        "MUJOCO_GL not set; defaulting to %r for headless MuJoCo rendering.", value
    )


def log_intrinsic_norm_hint(method: str, outputs_normalized: bool) -> None:
    """One-time hint about how intrinsic rewards are being normalized."""
    if outputs_normalized:
        # When a module owns its own normalization we stay quiet by default;
        # callers may still choose to surface this at a higher level.
        return
    get_logger("intrinsic").info(
        "Intrinsic normalization: method=%r outputs are raw; applying trainer's global RunningRMS.",
        method,
    )


def log_domain_randomization(summary: str) -> None:
    """Informational message describing domain-randomization diagnostics."""
    get_logger("env").info(
        "Domain randomization applied on env.reset(): %s", summary
    )


# ---------------------------------------------------------------------------
# Metric logging (CSV + TensorBoard)
# ---------------------------------------------------------------------------


class TBLogger:
    """Thin wrapper around torch.utils.tensorboard.SummaryWriter."""

    def __init__(self, log_dir: Path) -> None:
        self._writer = None
        if SummaryWriter is None:
            raise RuntimeError(
                "TensorBoard SummaryWriter not available. Install 'tensorboard'."
            )
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, metrics: Mapping[str, float], step: int) -> None:
        assert self._writer is not None
        for k, v in metrics.items():
            try:
                self._writer.add_scalar(k, float(v), global_step=step)
            except Exception:
                # Be robust to non-float values sneaking in.
                pass

    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        assert self._writer is not None
        self._writer.add_text(tag, text, global_step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None


class CSVLogger:
    """Append-only CSV logger with stable header."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[list[str]] = None
        # If file is empty, we'll write header on first log.
        self._wrote_header = self.path.exists() and self.path.stat().st_size > 0

    def _ensure_writer(self, row: Mapping[str, object]) -> None:
        if self._writer is not None:
            return
        # Stable field order: "step" first, then sorted rest.
        keys = [k for k in row.keys() if k != "step"]
        self._fieldnames = ["step"] + sorted(keys)
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        if not self._wrote_header:
            self._writer.writeheader()
            self._file.flush()
            self._wrote_header = True

    def log_row(self, step: int, metrics: Mapping[str, object]) -> None:
        row = {"step": int(step)}
        # Copy only primitive scalars (float/int/str/bool); cast others to str for safety.
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
        self._file.flush()  # ensure visible on disk for tailing

    def close(self) -> None:
        try:
            self._file.flush()
        finally:
            self._file.close()


class MetricLogger:
    """Unified scalar logger: CSV on cadence, TB each call."""

    def __init__(self, run_dir: Path, cfg: LoggingConfig) -> None:
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.run_dir / "logs" / "scalars.csv"
        self.csv = CSVLogger(self.csv_path)

        self.tb: Optional[TBLogger] = None
        if bool(cfg.tb):
            tb_dir = self.run_dir / "tb"
            if SummaryWriter is None:
                # Downgrade gracefully: keep running with CSV only.
                self.tb = None
            else:
                self.tb = TBLogger(tb_dir)

        # Keep last step to gate CSV cadence.
        self._last_csv_write_step: Optional[int] = None

    # ------------- API -------------

    def log(self, step: int, **metrics: float) -> None:
        """Log floats by step.

        TB every call; CSV when cadence matches.
        """
        if self.tb is not None:
            self.tb.log_scalars(metrics, step)

        should_write_csv = False
        if self._last_csv_write_step is None or step != self._last_csv_write_step:
            if int(step) % int(self.cfg.csv_interval) == 0:
                should_write_csv = True

        if should_write_csv:
            self.csv.log_row(step, metrics)
            self._last_csv_write_step = int(step)

    def log_hparams(self, params: Mapping[str, object]) -> None:
        """Optional: snapshot hparams as text (TB only)."""
        if self.tb is None:
            return
        text_lines = [f"{k}: {v}" for k, v in params.items()]
        self.tb.add_text("hparams", "\n".join(text_lines), step=0)

    def close(self) -> None:
        try:
            self.csv.close()
        finally:
            if self.tb is not None:
                self.tb.close()
