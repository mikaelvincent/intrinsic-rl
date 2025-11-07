"""Running RMS normalizer for intrinsic rewards.

Tracks EMA of squared values and exposes:
- update(x): incorporate batch values
- normalize(x): divide by current RMS (sqrt(EMA[r^2]+eps))
- state_dict/load_state_dict: minimal serialization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RunningRMS:
    """Exponential running RMS: sqrt(EMA[r^2] + eps)."""

    beta: float = 0.99
    eps: float = 1e-8
    init: float = 1.0

    def __post_init__(self) -> None:
        # Store EMA over squared values r^2
        self._r2_ema: float = float(self.init) ** 2

    # ---------------- API ----------------

    def update(self, x: Any) -> None:
        """Update running EMA from a batch of values (arraylike)."""
        arr = np.asarray(x)
        if arr.size == 0:
            return
        # Convert to float64 for numeric robustness on long runs
        r2_mean = float(np.mean(np.square(arr, dtype=np.float64)))
        self._r2_ema = float(self.beta) * self._r2_ema + (1.0 - float(self.beta)) * r2_mean

    @property
    def rms(self) -> float:
        """Current RMS."""
        return float(np.sqrt(self._r2_ema + float(self.eps)))

    def normalize(self, x: Any) -> np.ndarray:
        """Return x / rms as float32 (shape preserved)."""
        arr = np.asarray(x, dtype=np.float32)
        denom = self.rms
        if not np.isfinite(denom) or denom <= 0.0:
            denom = 1.0
        return arr / denom

    # -------------- (De)Serialization ---------------

    def state_dict(self) -> dict:
        return {"r2_ema": float(self._r2_ema), "beta": float(self.beta), "eps": float(self.eps)}

    def load_state_dict(self, state: dict) -> None:
        self._r2_ema = float(state.get("r2_ema", self._r2_ema))
        self.beta = float(state.get("beta", self.beta))
        self.eps = float(state.get("eps", self.eps))
