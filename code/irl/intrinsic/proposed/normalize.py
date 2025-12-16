from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from irl.intrinsic.normalization import RunningRMS


@dataclass
class ComponentRMS:
    impact: RunningRMS
    lp: RunningRMS

    def update(self, impact_vals: np.ndarray | list[float], lp_vals: np.ndarray | list[float]) -> None:
        self.impact.update(impact_vals)
        self.lp.update(lp_vals)

    def normalize(self, impact_vals: np.ndarray, lp_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.impact.normalize(impact_vals), self.lp.normalize(lp_vals)
