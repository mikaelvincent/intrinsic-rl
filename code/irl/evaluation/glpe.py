from __future__ import annotations

import numpy as np
import torch


def glpe_gate_and_intrinsic_no_update(
    mod: object, obs_1d: torch.Tensor, next_obs_1d: torch.Tensor
) -> tuple[int, float] | None:
    if not (hasattr(mod, "icm") and hasattr(mod, "store")):
        return None
    stats = getattr(mod, "_stats", None)
    if not isinstance(stats, dict):
        return None
    try:
        with torch.no_grad():
            b_obs = obs_1d.unsqueeze(0)
            b_next = next_obs_1d.unsqueeze(0)

            phi_t = mod.icm._phi(b_obs)
            rid = int(mod.store.locate(phi_t.detach().cpu().numpy().reshape(-1)))

            st = stats.get(rid)
            gate = 1 if st is None else int(getattr(st, "gate", 1))

            lp_raw = 0.0
            if st is not None and int(getattr(st, "count", 0)) > 0:
                lp_raw = max(
                    0.0,
                    float(getattr(st, "ema_long", 0.0) - getattr(st, "ema_short", 0.0)),
                )

            impact_raw_t = mod._impact_per_sample(b_obs, b_next)
            impact_raw = float(impact_raw_t.view(-1)[0].item())

            a_imp = float(getattr(mod, "alpha_impact", 1.0))
            a_lp = float(getattr(mod, "alpha_lp", 0.0))

            if bool(getattr(mod, "_normalize_inside", False)) and hasattr(mod, "_rms"):
                imp_n, lp_n = mod._rms.normalize(
                    np.asarray([impact_raw], dtype=np.float32),
                    np.asarray([lp_raw], dtype=np.float32),
                )
                combined = a_imp * float(imp_n[0]) + a_lp * float(lp_n[0])
            else:
                combined = a_imp * float(impact_raw) + a_lp * float(lp_raw)

            return int(gate), float(gate) * float(combined)
    except Exception:
        return None
