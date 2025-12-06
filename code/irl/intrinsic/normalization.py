"""Running RMS normalizer for intrinsic rewards.

Tracks an exponential moving average (EMA) of squared values and exposes helpers
to update the accumulator, query the current RMS, and serialise/restore state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RunningRMS:
    """Exponential running RMS normaliser.

    The normaliser tracks an EMA over squared values :math:`r^2` and exposes
    helpers to update the accumulator and normalise new batches by the current
    root-mean-square.

    Parameters
    ----------
    beta : float
        Exponential decay factor for the EMA over squared values. Values closer
        to ``1.0`` give a slower, more stable estimate.
    eps : float
        Small constant added under the square root to guard against numerical
        issues when the accumulator is close to zero.
    init : float
        Initial RMS estimate used before any calls to :meth:`update`.
    """

    beta: float = 0.99
    eps: float = 1e-8
    init: float = 1.0

    def __post_init__(self) -> None:
        # Store EMA over squared values r^2
        self._r2_ema: float = float(self.init) ** 2

    # ---------------- API ----------------

    def update(self, x: Any) -> None:
        """Update the running EMA from a batch of values.

        Parameters
        ----------
        x : array-like
            Values whose squared magnitude contributes to the running
            :math:`r^2` estimate. Any shape is accepted; only the mean of
            squared entries is used.
        """
        arr = np.asarray(x)
        if arr.size == 0:
            return
        # Convert to float64 for numeric robustness on long runs
        r2_mean = float(np.mean(np.square(arr, dtype=np.float64)))
        self._r2_ema = float(self.beta) * self._r2_ema + (1.0 - float(self.beta)) * r2_mean

    @property
    def rms(self) -> float:
        """Return the current RMS estimate."""
        return float(np.sqrt(self._r2_ema + float(self.eps)))

    def normalize(self, x: Any) -> np.ndarray:
        """Normalise values by the current RMS.

        Parameters
        ----------
        x : array-like
            Values to normalise. The input is converted to ``float32``.

        Returns
        -------
        numpy.ndarray
            Normalised values with the same shape as ``x`` and ``dtype=float32``.
        """
        arr = np.asarray(x, dtype=np.float32)
        denom = self.rms
        if not np.isfinite(denom) or denom <= 0.0:
            denom = 1.0
        return arr / denom

    # -------------- (De)Serialization ---------------

    def state_dict(self) -> dict:
        """Return a serialisable snapshot of the normaliser state."""
        return {"r2_ema": float(self._r2_ema), "beta": float(self.beta), "eps": float(self.eps)}

    def load_state_dict(self, state: dict) -> None:
        """Load state previously produced by :meth:`state_dict`."""
        self._r2_ema = float(state.get("r2_ema", self._r2_ema))
        self.beta = float(state.get("beta", self.beta))
        self.eps = float(state.get("eps", self.eps))
