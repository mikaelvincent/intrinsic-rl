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

    def update_scalar(self, impact_val: float, lp_val: float) -> None:
        self.impact.update_scalar(float(impact_val))
        self.lp.update_scalar(float(lp_val))

    def normalize(self, impact_vals: np.ndarray, lp_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.impact.normalize(impact_vals), self.lp.normalize(lp_vals)

    def normalize_scalar(self, impact_val: float, lp_val: float) -> tuple[float, float]:
        return (
            self.impact.normalize_scalar(float(impact_val)),
            self.lp.normalize_scalar(float(lp_val)),
        )

    def state_dict(self) -> dict:
        return {"impact": self.impact.state_dict(), "lp": self.lp.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        impact = state.get("impact")
        lp = state.get("lp")
        if isinstance(impact, dict):
            self.impact.load_state_dict(impact)
        if isinstance(lp, dict):
            self.lp.load_state_dict(lp)
