from __future__ import annotations

import numpy as np


class RunningObsNorm:
    def __init__(self, shape: int):
        self.count = 0.0
        self.mean = np.zeros((shape,), dtype=np.float64)
        self.var = np.ones((shape,), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        b = float(x.shape[0])
        if b == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)

        if self.count == 0.0:
            self.mean = batch_mean
            self.var = batch_var
            self.count = b
            return

        delta = batch_mean - self.mean
        tot = self.count + b
        new_mean = self.mean + delta * (b / tot)

        m_a = self.var * self.count
        m_b = batch_var * b
        new_var = (m_a + m_b + (delta**2) * (self.count * b / tot)) / tot

        self.mean = np.maximum(new_mean, -1e12)
        self.var = np.maximum(new_var, 1e-12)
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + 1e-8)
        return (x - self.mean) / std

    def state_dict(self) -> dict:
        return {"count": float(self.count), "mean": self.mean, "var": self.var}
