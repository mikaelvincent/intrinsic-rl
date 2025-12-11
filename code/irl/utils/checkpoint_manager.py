"""Checkpoint manager with atomic writes and optional retention.

This module provides the concrete implementation of:

* A :class:`CheckpointManager` that periodically writes torch
  checkpoints, maintains a ``ckpt_latest.pt`` alias, and prunes older
  step-numbered checkpoints.
* A compatibility layer for torch.load across versions.
* A convenience ``load_checkpoint`` wrapper.

Higher-level code should normally import these via :mod:`irl.utils.checkpoint`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .atomic_files import atomic_replace

_CKPT_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")


def _safe_torch_save(obj: Any, path: Path) -> None:
    """Write a torch payload via a temporary file, then replace.

    This reduces the risk of visible partial files if the process is
    interrupted mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    atomic_replace(tmp, path)


def _torch_load_compat(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load a checkpoint in a way that is compatible across torch versions.

    Parameters
    ----------
    path :
        Path to the checkpoint file.
    map_location :
        Argument forwarded to :func:`torch.load`.

    Returns
    -------
    dict
        Deserialised checkpoint payload.

    Notes
    -----
    Torch 2.6 adds a ``weights_only`` argument to :func:`torch.load`.
    This helper always opts out of that flag so a single call site
    works for both older and newer versions.
    """
    try:
        # torch >= 2.6
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # torch < 2.6 (no weights_only kwarg)
        return torch.load(path, map_location=map_location)


@dataclass
class CheckpointManager:
    """Manage periodic saves, pruning, and a ``ckpt_latest.pt`` alias.

    Parameters
    ----------
    run_dir :
        Root directory for a single training or evaluation run.
    interval_steps :
        Save cadence expressed in environment steps.
    max_to_keep :
        Maximum number of step-numbered checkpoints to retain. The
        ``ckpt_latest.pt`` alias is always preserved.
    subdir :
        Subdirectory under ``run_dir`` where checkpoints are stored.
    """

    run_dir: Path
    interval_steps: int
    max_to_keep: Optional[int] = 5
    subdir: str = "checkpoints"

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.ckpt_dir = self.run_dir / self.subdir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Track the most recent checkpoint step so cadence logic can detect
        # when an interval boundary has been crossed between calls to
        # ``should_save``. This is initialised from disk when possible.
        self._last_saved_step: Optional[int] = self._discover_last_saved_step()

    # ------------- Paths -------------

    @property
    def latest_path(self) -> Path:
        """Return the path to the ``ckpt_latest.pt`` alias."""
        return self.ckpt_dir / "ckpt_latest.pt"

    def path_for_step(self, step: int) -> Path:
        """Return the canonical checkpoint path for a given step."""
        return self.ckpt_dir / f"ckpt_step_{int(step)}.pt"

    # ------------- Save/Load -------------

    def save(self, step: int, payload: Dict[str, Any]) -> Path:
        """Save a checkpoint payload and update the ``latest`` alias.

        Parameters
        ----------
        step :
            Global step associated with the checkpoint.
        payload :
            Mapping to serialise with :func:`torch.save`. A ``"step"``
            key is added or overwritten to match ``step``.

        Returns
        -------
        pathlib.Path
            The path of the step-numbered checkpoint.
        """
        step_int = int(step)
        if "step" not in payload:
            payload = dict(payload)
            payload["step"] = step_int

        path = self.path_for_step(step_int)
        _safe_torch_save(payload, path)

        # Update latest alias.
        _safe_torch_save(payload, self.latest_path)

        # Update in-memory cadence tracker so subsequent calls to
        # ``should_save`` use this step as the last checkpoint boundary.
        self._last_saved_step = step_int

        # Prune old checkpoints (except 'latest')
        self._prune_old()
        return path

    def should_save(self, step: int) -> bool:
        """Return ``True`` when a new checkpoint should be written.

        The manager requests a checkpoint whenever:

        * the configured interval is positive,
        * the current step is strictly greater than the last checkpoint
          step, and
        * the difference ``step - last_saved_step`` is at least
          ``interval_steps``.

        This threshold-based rule ensures that interval boundaries crossed
        inside long rollouts are honoured at the next call site instead of
        relying on ``step % interval == 0``.
        """
        if self.interval_steps <= 0:
            return False

        s = int(step)
        last = self._last_saved_step if self._last_saved_step is not None else 0

        # Never request a save twice for the same (or a smaller) step.
        if s <= last:
            return False

        return (s - last) >= int(self.interval_steps)

    def load_latest(self, map_location: str | torch.device = "cpu") -> Tuple[Dict[str, Any], int]:
        """Load the latest checkpoint payload and its associated step.

        Parameters
        ----------
        map_location :
            Argument forwarded to :func:`torch.load`.

        Returns
        -------
        tuple
            ``(payload, step)`` where ``payload`` is the deserialised
            dict and ``step`` is the integer training step inferred from
            the payload or filename.

        Raises
        ------
        FileNotFoundError
            If no ``ckpt_latest.pt`` file is present.
        """
        if not self.latest_path.exists():
            raise FileNotFoundError(f"No latest checkpoint at {self.latest_path}")
        payload = _torch_load_compat(self.latest_path, map_location=map_location)
        step = int(
            payload.get(
                "step",
                self._infer_step_from_name(self.latest_path.name) or -1,
            )
        )
        return payload, step

    # ------------- Helpers -------------

    def _prune_old(self) -> None:
        """Remove older step-numbered checkpoints beyond ``max_to_keep``."""
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
                # Best-effort pruning: ignore deletion failures.
                pass

    @staticmethod
    def _infer_step_from_name(name: str) -> Optional[int]:
        """Extract the integer step from a ``ckpt_step_*.pt`` filename."""
        m = _CKPT_RE.match(name)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _discover_last_saved_step(self) -> Optional[int]:
        """Best-effort detection of the last checkpoint step on disk.

        Preference is given to ``ckpt_latest.pt`` when present; if it
        cannot be loaded, the method falls back to scanning step-numbered
        checkpoint filenames. Returns ``None`` when no checkpoints are
        found.
        """
        # Prefer the latest alias: it should contain a 'step' entry.
        if self.latest_path.exists():
            try:
                payload = _torch_load_compat(self.latest_path, map_location="cpu")
                step = payload.get("step")
                if isinstance(step, (int, float)):
                    return int(step)
            except Exception:
                # Ignore and fall back to filename-based discovery.
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
    """Convenience wrapper for loading a single checkpoint file."""
    return _torch_load_compat(Path(path), map_location=map_location)
