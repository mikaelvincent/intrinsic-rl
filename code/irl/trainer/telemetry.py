from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class ProgressLogger:
    start_wall: float
    last_log_time: float
    last_log_step: int

    @classmethod
    def start(cls, *, initial_step: int) -> "ProgressLogger":
        now = time.time()
        return cls(start_wall=now, last_log_time=now, last_log_step=int(initial_step))

    def maybe_log(
        self,
        *,
        logger: object,
        update_idx: int,
        global_step: int,
        total_steps: int,
        log_every_updates: int,
        log_payload: Mapping[str, object],
        t_update_total: float,
        t_rollout_total: float,
        t_rollout_policy: float,
        t_rollout_env_step: float,
        t_rollout_intrinsic_step: float,
        t_rollout_other: float,
        t_intrinsic_compute: float,
        t_intrinsic_update: float,
        t_gae: float,
        t_ppo: float,
        t_logging_compute: float,
        t_ml_log: float,
    ) -> None:
        if (int(update_idx) % int(log_every_updates) != 0) and int(global_step) < int(total_steps):
            return

        now = time.time()
        elapsed = max(now - float(self.start_wall), 1e-6)
        avg_sps = float(global_step) / elapsed if int(global_step) > 0 else 0.0
        delta_steps = int(global_step - int(self.last_log_step))
        delta_t = max(now - float(self.last_log_time), 1e-6)
        recent_sps = float(delta_steps) / delta_t if delta_steps > 0 else 0.0

        approx_kl = float(log_payload.get("approx_kl", float("nan")))
        clip_frac = float(log_payload.get("clip_frac", float("nan")))
        r_total = float(log_payload.get("reward_total_mean", float("nan")))
        r_int_mean = (
            float(log_payload.get("r_int_mean", 0.0)) if "r_int_mean" in log_payload else 0.0
        )

        getattr(logger, "info")(
            "Train progress: step=%d update=%d avg_sps=%.1f recent_sps=%.1f "
            "reward_total_mean=%.3f r_int_mean=%.3f approx_kl=%.4f clip_frac=%.3f",
            int(global_step),
            int(update_idx),
            avg_sps,
            recent_sps,
            r_total,
            r_int_mean,
            approx_kl,
            clip_frac,
        )

        intrinsic_total = float(t_rollout_intrinsic_step + t_intrinsic_compute + t_intrinsic_update)
        logging_total = float(t_logging_compute + t_ml_log)
        getattr(logger, "info")(
            "Timings (s) update=%d: total=%.3f | rollout=%.3f (policy=%.3f, env_step=%.3f, "
            "intr_step=%.3f, other=%.3f) | intrinsic_total=%.3f (batch_compute=%.3f, update=%.3f) | "
            "gae=%.3f | ppo=%.3f | logging=%.3f (compute=%.3f, io=%.3f)",
            int(update_idx),
            float(t_update_total),
            float(t_rollout_total),
            float(t_rollout_policy),
            float(t_rollout_env_step),
            float(t_rollout_intrinsic_step),
            float(t_rollout_other),
            intrinsic_total,
            float(t_intrinsic_compute),
            float(t_intrinsic_update),
            float(t_gae),
            float(t_ppo),
            logging_total,
            float(t_logging_compute),
            float(t_ml_log),
        )

        self.last_log_time = now
        self.last_log_step = int(global_step)
