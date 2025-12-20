from __future__ import annotations

from typing import Final

COL_STEP: Final[str] = "step"

COL_REWARD_MEAN: Final[str] = "reward_mean"
COL_REWARD_TOTAL_MEAN: Final[str] = "reward_total_mean"
COL_EPISODE_RETURN_MEAN: Final[str] = "episode_return_mean"

COL_GATE_RATE: Final[str] = "gate_rate"
COL_GATE_RATE_PCT: Final[str] = "gate_rate_pct"

COL_IMPACT_RMS: Final[str] = "impact_rms"
COL_LP_RMS: Final[str] = "lp_rms"

SCALARS_REQUIRED_COMMON_COLS: Final[tuple[str, ...]] = (
    COL_STEP,
    COL_REWARD_MEAN,
    COL_REWARD_TOTAL_MEAN,
    COL_EPISODE_RETURN_MEAN,
)

GLPE_GATE_ANY_COLS: Final[tuple[str, ...]] = (COL_GATE_RATE, COL_GATE_RATE_PCT)
GLPE_REQUIRED_COMPONENT_COLS: Final[tuple[str, ...]] = (COL_IMPACT_RMS, COL_LP_RMS)

REWARD_METRIC_FALLBACKS: Final[dict[str, tuple[str, ...]]] = {
    COL_REWARD_MEAN: (COL_REWARD_TOTAL_MEAN,),
    COL_REWARD_TOTAL_MEAN: (COL_REWARD_MEAN,),
}
