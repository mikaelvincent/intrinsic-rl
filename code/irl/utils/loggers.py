from __future__ import annotations

import csv
import json
import logging
import math
from logging import Logger
from numbers import Number
from pathlib import Path
from typing import Mapping, Optional

from irl.cfg.schema import LoggingConfig
from irl.utils.checkpoint import atomic_write_text, compute_cfg_hash

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
        self._dropped_keys_warned: set[str] = set()

    def _ensure_writer(self, row: Mapping[str, object]) -> None:
        if self._writer is not None:
            return
        keys = [k for k in row.keys() if k != "step"]
        self._fieldnames = ["step"] + sorted(keys)
        self._writer = csv.DictWriter(
            self._file, fieldnames=self._fieldnames, extrasaction="ignore"
        )
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

        if self._fieldnames is not None:
            extra = set(row.keys()) - set(self._fieldnames)
            if extra:
                newly_warned = extra - self._dropped_keys_warned
                if newly_warned:
                    get_logger("csv").warning(
                        "Dropping metrics not in CSV header for %s: %s",
                        self.path,
                        ", ".join(sorted(newly_warned)),
                    )
                    self._dropped_keys_warned |= newly_warned

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

        self._last_csv_write_step: Optional[int] = None
        self._last_finite: dict[str, float] = {}

    @staticmethod
    def _num(v: object) -> float | None:
        if isinstance(v, bool):
            return None
        if not isinstance(v, Number):
            return None
        try:
            return float(v)
        except Exception:
            return None

    def log(self, step: int, **metrics: object) -> bool:
        interval = int(max(1, self.cfg.csv_interval))
        s = int(step)
        last = self._last_csv_write_step

        for k, v in metrics.items():
            x = self._num(v)
            if x is None:
                continue
            if math.isfinite(x):
                self._last_finite[str(k)] = float(x)

        should_write_csv = False
        if last is None:
            if s == 0 or s >= interval:
                should_write_csv = True
        else:
            if s >= last + interval:
                should_write_csv = True

        if not should_write_csv:
            return False

        out: dict[str, object] = {}
        for k, v in metrics.items():
            kk = str(k)
            x = self._num(v)
            if x is None:
                out[kk] = v
                continue
            if math.isfinite(x):
                out[kk] = v
                continue
            if kk in self._last_finite:
                out[kk] = float(self._last_finite[kk])
            else:
                out[kk] = 0.0

        self.csv.log_row(s, out)
        self._last_csv_write_step = s
        return True

    def log_hparams(self, params: Mapping[str, object]) -> None:
        if not isinstance(params, Mapping):
            return

        cfg_path = self.run_dir / "config.json"
        hash_path = self.run_dir / "config_hash.txt"

        def _write_hash_file(h: str) -> None:
            hs = str(h).strip()
            if not hs:
                return
            try:
                if hash_path.exists() and hash_path.is_file() and hash_path.stat().st_size > 0:
                    return
            except Exception:
                return
            atomic_write_text(hash_path, hs + "\n")

        try:
            if cfg_path.exists() and cfg_path.is_file() and cfg_path.stat().st_size > 0:
                try:
                    existing = json.loads(cfg_path.read_text(encoding="utf-8"))
                except Exception:
                    return
                if isinstance(existing, Mapping):
                    h = existing.get("cfg_hash")
                    if not isinstance(h, str) or not h.strip():
                        tmp = dict(existing)
                        tmp.pop("cfg_hash", None)
                        h = compute_cfg_hash(tmp)
                    _write_hash_file(str(h))
                return
        except Exception:
            return

        cfg_dict = dict(params)
        cfg_dict.pop("cfg_hash", None)
        cfg_hash = compute_cfg_hash(cfg_dict)

        payload = dict(cfg_dict)
        payload["cfg_hash"] = cfg_hash

        atomic_write_text(cfg_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        _write_hash_file(cfg_hash)

    def close(self) -> None:
        self.csv.close()
