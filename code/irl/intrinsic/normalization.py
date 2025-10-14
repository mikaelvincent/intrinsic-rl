"""Running RMS normalization utility for intrinsic rewards.

This module provides a lightweight exponential moving average (EMA) of the
squared values to estimate a running root-mean-square (RMS). It is intended to
stabilize intrinsic reward scales across time and across intrinsic modules.

Typical usage
-------------
>>> rms = RunningRMS(beta=0.99)
>>> rms.update(r_vec)                 # update from a batch of raw intrinsic values
>>> r_norm = rms.normalize(r_vec)     # divide by current RMS (with eps guard)
>>> current = rms.rms                 # scalar RMS value
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RunningRMS:
    """Exponential running RMS: sqrt(EMA[r^2] + eps).

    Parameters
    ----------
    beta : float
        EMA coefficient in [0, 1). Larger -> slower updates. Default 0.99.
    eps : float
        Numerical stability term added under the square root.
    init : float
        Initial RMS value. Internally we initialize EMA of r^2 to `init**2`.
    """

    beta: float = 0.99
    eps: float = 1e-8
    init: float = 1.0

    def __post_init__(self) -> None:
        # Store EMA over squared values r^2
        self._r2_ema: float = float(self.init) ** 2

    # ---------------- API ----------------

    def update(self, x: Any) -> None:
        """Update running EMA from a batch of values.

        Accepts arraylike; empty inputs are ignored.
        """
        arr = np.asarray(x)
        if arr.size == 0:
            return
        # Convert to float64 for numeric robustness on long runs
        r2_mean = float(np.mean(np.square(arr, dtype=np.float64)))
        self._r2_ema = float(self.beta) * self._r2_ema + (1.0 - float(self.beta)) * r2_mean

    @property
    def rms(self) -> float:
        """Current RMS = sqrt(EMA[r^2] + eps)."""
        return float(np.sqrt(self._r2_ema + float(self.eps)))

    def normalize(self, x: Any) -> np.ndarray:
        """Return x / rms as a float32 numpy array (shape preserved)."""
        arr = np.asarray(x, dtype=np.float32)
        denom = self.rms
        # Avoid division by zero even if eps or init were misconfigured
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
