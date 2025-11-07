"""Checkpoint manager with atomic writes & bounded retention.

See devspec/dev_spec_and_plan.md §6.1 (artifacts) and §9 (reliability).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


_CKPT_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")


def _atomic_replace(src: Path, dst: Path) -> None:
    """Atomic move where supported; falls back to replace."""
    os.replace(src, dst)


def _safe_torch_save(obj: Any, path: Path) -> None:
    """Write via tmp file then replace to reduce corruption risk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    _atomic_replace(tmp, path)


def _torch_load_compat(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load payload across torch versions (opt out of weights_only)."""
    try:
        # torch >= 2.6
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # torch < 2.6 (no weights_only kwarg)
        return torch.load(path, map_location=map_location)


@dataclass
class CheckpointManager:
    """Manage periodic saves, pruning, and 'latest' symlink-equivalent."""

    run_dir: Path
    interval_steps: int
    max_to_keep: Optional[int] = 5
    subdir: str = "checkpoints"

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.ckpt_dir = self.run_dir / self.subdir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------- Paths -------------

    @property
    def latest_path(self) -> Path:
        return self.ckpt_dir / "ckpt_latest.pt"

    def path_for_step(self, step: int) -> Path:
        return self.ckpt_dir / f"ckpt_step_{int(step)}.pt"

    # ------------- Save/Load -------------

    def save(self, step: int, payload: Dict[str, Any]) -> Path:
        """Save payload at a specific step and update 'latest'."""
        if "step" not in payload:
            payload = dict(payload)
            payload["step"] = int(step)

        path = self.path_for_step(step)
        _safe_torch_save(payload, path)

        # Update latest
        _safe_torch_save(payload, self.latest_path)

        # Prune old checkpoints (except 'latest')
        self._prune_old()
        return path

    def should_save(self, step: int) -> bool:
        return int(step) % int(self.interval_steps) == 0

    def load_latest(self, map_location: str | torch.device = "cpu") -> Tuple[Dict[str, Any], int]:
        """Load latest checkpoint (returns payload, step)."""
        if not self.latest_path.exists():
            raise FileNotFoundError(f"No latest checkpoint at {self.latest_path}")
        payload = _torch_load_compat(self.latest_path, map_location=map_location)
        step = int(payload.get("step", self._infer_step_from_name(self.latest_path.name) or -1))
        return payload, step

    # ------------- Helpers -------------

    def _prune_old(self) -> None:
        if self.max_to_keep is None or self.max_to_keep <= 0:
            return

        ckpts = []
        for p in self.ckpt_dir.iterdir():
            if p.is_file() and _CKPT_RE.match(p.name):
                step = self._infer_step_from_name(p.name)
                if step is not None:
                    ckpts.append((step, p))
        ckpts.sort(key=lambda t: t[0], reverse=True)

        for _, path in ckpts[self.max_to_keep :]:
            try:
                path.unlink()
            except Exception:
                pass  # ignore removal failures

    @staticmethod
    def _infer_step_from_name(name: str) -> Optional[int]:
        m = _CKPT_RE.match(name)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Convenience free function for direct loads."""
    return _torch_load_compat(path, map_location=map_location)
