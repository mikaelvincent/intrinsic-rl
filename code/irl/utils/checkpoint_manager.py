from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .atomic_files import atomic_replace

_CKPT_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")
_WARMUP_CHECKPOINTS = 10


def _safe_torch_save(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    atomic_replace(tmp, path)


def _torch_load_compat(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    # Torch 2.6 adds weights_only; opt out for full payload compatibility.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class CheckpointManager:
    run_dir: Path
    interval_steps: int
    max_to_keep: Optional[int] = 5
    subdir: str = "checkpoints"

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.ckpt_dir = self.run_dir / self.subdir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._last_saved_step: Optional[int] = self._discover_last_saved_step()
        self._warmup_last_idx: int = -1
        self._regular_last_step: int = 0
        self._init_cadence_state_from_disk()

    @property
    def latest_path(self) -> Path:
        return self.ckpt_dir / "ckpt_latest.pt"

    def path_for_step(self, step: int) -> Path:
        return self.ckpt_dir / f"ckpt_step_{int(step)}.pt"

    def save(self, step: int, payload: Dict[str, Any]) -> Path:
        step_int = int(step)
        if "step" not in payload:
            payload = dict(payload)
            payload["step"] = step_int

        path = self.path_for_step(step_int)
        _safe_torch_save(payload, path)
        _safe_torch_save(payload, self.latest_path)

        self._last_saved_step = step_int
        self._update_cadence_after_save(step_int)
        self._prune_old()
        return path

    def should_save(self, step: int) -> bool:
        if self.interval_steps <= 0:
            return False

        s = int(step)
        last_any = int(self._last_saved_step) if self._last_saved_step is not None else -1
        if s <= last_any:
            return False

        N = int(self.interval_steps)

        # Warmup: up to 10 evenly spaced checkpoints for steps in [0, N).
        if s < N:
            next_idx = self._warmup_last_idx + 1
            if next_idx < _WARMUP_CHECKPOINTS:
                return s >= self._warmup_target_step(next_idx)
            return False

        # Regular cadence is independent of warmup saves.
        return (s - int(self._regular_last_step)) >= N

    def load_latest(self, map_location: str | torch.device = "cpu") -> Tuple[Dict[str, Any], int]:
        if not self.latest_path.exists():
            raise FileNotFoundError(f"No latest checkpoint at {self.latest_path}")
        payload = _torch_load_compat(self.latest_path, map_location=map_location)
        step = int(payload.get("step", self._infer_step_from_name(self.latest_path.name) or -1))
        return payload, step

    def _warmup_target_step(self, idx: int) -> int:
        N = int(self.interval_steps)
        if N <= 0:
            return 0
        i = int(max(0, min(idx, _WARMUP_CHECKPOINTS - 1)))
        return (i * N) // _WARMUP_CHECKPOINTS

    def _warmup_idx_for_step(self, step: int) -> int:
        N = int(self.interval_steps)
        if N <= 0:
            return 0
        s = int(max(0, min(step, N - 1)))
        idx = (s * _WARMUP_CHECKPOINTS) // N
        return int(max(0, min(idx, _WARMUP_CHECKPOINTS - 1)))

    def _update_cadence_after_save(self, step: int) -> None:
        N = int(self.interval_steps)
        s = int(step)
        if N <= 0:
            return

        if s < N:
            self._warmup_last_idx = max(self._warmup_last_idx, self._warmup_idx_for_step(s))
            return

        self._regular_last_step = s
        self._warmup_last_idx = _WARMUP_CHECKPOINTS - 1

    def _init_cadence_state_from_disk(self) -> None:
        N = int(self.interval_steps)
        last = self._last_saved_step

        if N <= 0 or last is None:
            self._regular_last_step = 0
            self._warmup_last_idx = -1
            return

        s = int(last)
        if s < N:
            self._regular_last_step = 0
            self._warmup_last_idx = self._warmup_idx_for_step(s)
            return

        self._regular_last_step = s
        self._warmup_last_idx = _WARMUP_CHECKPOINTS - 1

    def _prune_old(self) -> None:
        if self.max_to_keep is None or self.max_to_keep <= 0:
            return

        ckpts: list[tuple[int, Path]] = []
        for p in self.ckpt_dir.iterdir():
            if not (p.is_file() and _CKPT_RE.match(p.name)):
                continue
            step = self._infer_step_from_name(p.name)
            if step is None or int(step) == 0:
                continue
            ckpts.append((int(step), p))

        ckpts.sort(key=lambda t: t[0], reverse=True)
        for _, path in ckpts[int(self.max_to_keep) :]:
            try:
                path.unlink()
            except Exception:
                pass

    @staticmethod
    def _infer_step_from_name(name: str) -> Optional[int]:
        m = _CKPT_RE.match(name)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _discover_last_saved_step(self) -> Optional[int]:
        if self.latest_path.exists():
            try:
                payload = _torch_load_compat(self.latest_path, map_location="cpu")
                step = payload.get("step")
                if isinstance(step, (int, float)):
                    return int(step)
            except Exception:
                pass

        max_step: Optional[int] = None
        if self.ckpt_dir.exists():
            for p in self.ckpt_dir.iterdir():
                if not p.is_file():
                    continue
                s = self._infer_step_from_name(p.name)
                if s is None:
                    continue
                if max_step is None or s > max_step:
                    max_step = s
        return max_step


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return _torch_load_compat(Path(path), map_location=map_location)
