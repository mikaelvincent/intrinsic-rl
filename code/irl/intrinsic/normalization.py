from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RunningRMS:
    beta: float = 0.99
    eps: float = 1e-8
    init: float = 1.0

    def __post_init__(self) -> None:
        self._r2_ema: float = float(self.init) ** 2

    def update(self, x: Any) -> None:
        arr = np.asarray(x)
        if arr.size == 0:
            return
        r2_mean = float(np.mean(np.square(arr, dtype=np.float64)))
        self._r2_ema = float(self.beta) * self._r2_ema + (1.0 - float(self.beta)) * r2_mean

    def update_scalar(self, x: float) -> None:
        v = float(x)
        r2_mean = float(v * v)
        self._r2_ema = float(self.beta) * self._r2_ema + (1.0 - float(self.beta)) * r2_mean

    @property
    def rms(self) -> float:
        return float(np.sqrt(self._r2_ema + float(self.eps)))

    def normalize(self, x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        denom = self.rms
        if not np.isfinite(denom) or denom <= 0.0:
            denom = 1.0
        return arr / denom

    def normalize_scalar(self, x: float) -> float:
        v = float(x)
        denom = self.rms
        if not np.isfinite(denom) or denom <= 0.0:
            denom = 1.0
        return float(v) / float(denom)

    def state_dict(self) -> dict:
        return {"r2_ema": float(self._r2_ema), "beta": float(self.beta), "eps": float(self.eps)}

    def load_state_dict(self, state: dict) -> None:
        self._r2_ema = float(state.get("r2_ema", self._r2_ema))
        self.beta = float(state.get("beta", self.beta))
        self.eps = float(state.get("eps", self.eps))
