"""R-IAC intrinsic module (per-region EMAs and Learning Progress).

Implements the core of Sprint 3 — Step 2:
* Maintain **per-region** exponential moving averages (EMAs) of forward-model
  error in the learned embedding space φ(s).
* Compute region-wise **Learning Progress**:
      LP_i = max(0, EMA_long_i - EMA_short_i)
  which is positive when the error trends downward (recent < long-term).

**Sprint 3 — Step 3 (this change):**
* Intrinsic is now *normalized* online via a per-module running RMS over LP:
      r_int = α_LP * normalize(LP_i)
  Normalization uses a simple exponential running RMS (EMA of r^2). The module
  advertises `outputs_normalized=True` so higher-level trainers can skip an
  additional global normalization to avoid double normalization.

**Sprint 3 — Step 4 (this change):**
* Diagnostics export helpers:
    - `export_diagnostics(out_dir: Path, step: int)` appends:
        • `regions.jsonl` — one JSON object per region with stats and bbox
        • `gates.csv`     — rows (step, region_id, gate); gates=1 initially

Design
------
- We reuse the ICM encoder and forward head to obtain φ(s), φ(s') and a
  forward prediction φ̂(s') given (φ(s), a). Error is computed as the mean
  squared difference between φ̂(s') and φ(s').
- Regions are maintained by a KD-tree (`KDTreeRegionStore`) over φ-space.
  Each inserted φ(s_t) returns a (leaf) region id; we update that region's
  EMAs with the current transition's error.
- Intrinsic for a transition equals `alpha_lp * normalize(LP(region_of(φ(s_t))))`.
  (Trainer-level η scaling and clipping are still applied outside this module.)

Public API (aligned with other intrinsic modules)
------------------------------------------------
- compute(tr) -> IntrinsicOutput
- compute_batch(obs, next_obs, actions, reduction="none") -> Tensor[[B] or scalar]
- loss(obs, next_obs, actions) -> dict(total, icm_forward, icm_inverse, intrinsic_mean)
- update(obs, next_obs, actions, steps=1) -> dict(loss_total, loss_forward, loss_inverse, intrinsic_mean)
- export_diagnostics(out_dir: Path, step: int) -> None   [new in Sprint 3 — Step 4]

Notes
-----
- Gating and adaptive weighting are *not* implemented here (Sprint 4).
- Region stats are kept in Python dicts and are not part of the state_dict
  (lightweight; fine for this sprint).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union
import csv
import json

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

import gymnasium as gym

from . import BaseIntrinsicModule, IntrinsicOutput, Transition
from .icm import ICM, ICMConfig
from .regions import KDTreeRegionStore
from .normalization import RunningRMS
from irl.utils.torchops import as_tensor, ensure_2d


# ------------------------------ Small helpers ------------------------------


@dataclass
class _RegionStats:
    """Per-region EMA container."""

    ema_long: float = 0.0
    ema_short: float = 0.0
    count: int = 0


def simulate_lp_emas(errors: Iterable[float], beta_long: float = 0.995, beta_short: float = 0.90):
    """Utility for tests/diagnostics: run EMAs over a sequence of errors.

    Returns:
        (ema_long_seq, ema_short_seq, lp_seq) — lists of floats.
    """
    el, es, lp = [], [], []
    l = s = None
    for e in errors:
        x = float(e)
        if l is None:
            l = x
            s = x
        else:
            l = beta_long * l + (1.0 - beta_long) * x
            s = beta_short * s + (1.0 - beta_short) * x
        el.append(l)
        es.append(s)
        lp.append(max(0.0, l - s))
    return el, es, lp


# ---------------------------------- RIAC ------------------------------------


class RIAC(BaseIntrinsicModule, nn.Module):
    """R‑IAC: region-wise learning progress intrinsic reward (normalized).

    Args:
        obs_space: Box observation space (vector states).
        act_space: Discrete or Box action space (for ICM inverse/forward).
        device: torch device string or device.
        icm: Optionally supply an external ICM to share encoder/forward.
        icm_cfg: Optional ICMConfig for internal ICM if `icm` is None.
        region_capacity: KD-tree leaf capacity before split.
        depth_max: Maximum KD-tree depth.
        ema_beta_long / ema_beta_short: EMA coefficients.
        alpha_lp: Scaling applied to **normalized** LP before trainer-level η / clip.

    Notes:
        * This module keeps *no* learnable parameters besides the underlying ICM.
        * Outputs are **already normalized** via a per-module running RMS over LP.
          Trainers should check `outputs_normalized` and avoid renormalizing.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        device: Union[str, torch.device] = "cpu",
        icm: Optional[ICM] = None,
        icm_cfg: Optional[ICMConfig] = None,
        *,
        region_capacity: int = 200,
        depth_max: int = 12,
        ema_beta_long: float = 0.995,
        ema_beta_short: float = 0.90,
        alpha_lp: float = 0.5,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("RIAC currently supports Box observation spaces (vector states).")
        self.device = torch.device(device)

        # Backing ICM (provides encoder φ and forward head)
        self.icm = icm if icm is not None else ICM(obs_space, act_space, device=device, cfg=icm_cfg)
        self.encoder = self.icm.encoder  # shared weights
        self.is_discrete = self.icm.is_discrete
        self.obs_dim = int(obs_space.shape[0])
        self.phi_dim = int(self.icm.cfg.phi_dim)

        # Regionization in φ-space
        self.store = KDTreeRegionStore(
            dim=self.phi_dim, capacity=int(region_capacity), depth_max=int(depth_max)
        )
        self._stats: dict[int, _RegionStats] = {}

        # EMA/scale knobs
        self.beta_long = float(ema_beta_long)
        self.beta_short = float(ema_beta_short)
        self.alpha_lp = float(alpha_lp)

        # NEW: per-module running RMS over LP values (for normalization)
        self._lp_rms = RunningRMS(beta=0.99, eps=1e-8)
        # Signal to trainers that this module returns normalized outputs
        self.outputs_normalized: bool = True

        self.to(self.device)

    # ----------------------- internal: per-region stats -----------------------

    def _update_region(self, rid: int, error: float) -> float:
        """Update EMAs for the region and return its current LP (>=0)."""
        st = self._stats.get(rid)
        if st is None or st.count == 0:
            self._stats[rid] = _RegionStats(ema_long=error, ema_short=error, count=1)
            return 0.0  # first sample: LP=0 by definition
        # EMA updates
        st.ema_long = self.beta_long * st.ema_long + (1.0 - self.beta_long) * error
        st.ema_short = self.beta_short * st.ema_short + (1.0 - self.beta_short) * error
        st.count += 1
        return max(0.0, st.ema_long - st.ema_short)

    # -------------------------- intrinsic computation -------------------------

    @torch.no_grad()
    def _forward_error_per_sample(
        self, obs: Tensor, next_obs: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Return per-sample forward error in φ space (mean over dims), shape [B]."""
        o = ensure_2d(obs)
        op = ensure_2d(next_obs)
        a = actions
        phi_t = self.icm._phi(o)
        phi_tp1 = self.icm._phi(op)
        a_fwd = self.icm._act_for_forward(a)
        pred = self.icm.forward_head(torch.cat([phi_t, a_fwd], dim=-1))
        per_dim = F.mse_loss(pred, phi_tp1, reduction="none")  # [B, D]
        return per_dim.mean(dim=-1), phi_t  # [B], [B, D]

    def compute(self, tr: Transition) -> IntrinsicOutput:
        """Compute α_LP * normalize(LP(region(φ(s)))) for a single transition."""
        with torch.no_grad():
            s = as_tensor(tr.s, self.device)
            sp = as_tensor(tr.s_next, self.device)
            a = as_tensor(tr.a, self.device)
            err, phi_t = self._forward_error_per_sample(
                s.view(1, -1), sp.view(1, -1), a.view(1, -1 if not self.is_discrete else 1)
            )
            # Insert φ(s) to update regionization and stats
            rid = int(self.store.insert(phi_t.detach().cpu().numpy().reshape(-1)))
            lp = self._update_region(rid, float(err.item()))

            # Normalize LP via running RMS, then scale by α_LP
            self._lp_rms.update([lp])
            lp_norm = self._lp_rms.normalize([lp])[0]
            r = self.alpha_lp * float(lp_norm)
            return IntrinsicOutput(r_int=float(r))

    @torch.no_grad()
    def compute_batch(
        self,
        obs: Any,
        next_obs: Any,
        actions: Any,
        reduction: str = "none",
    ) -> Tensor:
        """Vectorized α_LP * normalize(LP(region(φ(s)))) for a batch (updates EMAs)."""
        o = as_tensor(obs, self.device)
        op = as_tensor(next_obs, self.device)
        a = as_tensor(actions, self.device)

        # Forward-model error in φ-space
        err, phi_t = self._forward_error_per_sample(o, op, a)  # [B], [B,D]

        # Determine regions for φ(s) and update per-region EMAs
        phi_np = phi_t.detach().cpu().numpy()
        rids = self.store.bulk_insert(phi_np)  # np.ndarray [B]
        lp_vals: list[float] = []
        for i in range(err.shape[0]):
            lp = self._update_region(int(rids[i]), float(err[i].item()))
            lp_vals.append(lp)

        # Normalize LP via per-module RMS, then apply α_LP
        lp_arr = np.asarray(lp_vals, dtype=np.float32)
        self._lp_rms.update(lp_arr)
        lp_norm = self._lp_rms.normalize(lp_arr)  # np.ndarray [B]

        out = torch.as_tensor(self.alpha_lp * lp_norm, dtype=torch.float32, device=self.device)
        if reduction == "mean":
            return out.mean()
        return out

    # ----------------------------- losses & update ----------------------------

    def loss(self, obs: Any, next_obs: Any, actions: Any) -> dict[str, Tensor]:
        """Return ICM training loss + R‑IAC intrinsic mean (diagnostic only)."""
        icm_losses = self.icm.loss(obs, next_obs, actions)
        with torch.no_grad():
            o = as_tensor(obs, self.device)
            op = as_tensor(next_obs, self.device)
            a = as_tensor(actions, self.device)
            err, phi_t = self._forward_error_per_sample(o, op, a)
            # Use current per-region LPs without mutating state (locate only)
            rids = [self.store.locate(p) for p in phi_t.detach().cpu().numpy()]
            lp_vals = []
            for rid in rids:
                st = self._stats.get(int(rid))
                if st is None or st.count == 0:
                    lp_vals.append(0.0)
                else:
                    lp_vals.append(max(0.0, st.ema_long - st.ema_short))
            # Report normalized mean for diagnostics
            lp_arr = np.asarray(lp_vals, dtype=np.float32)
            lp_norm = self._lp_rms.normalize(lp_arr)
            r_mean = float(np.mean(self.alpha_lp * lp_norm)) if lp_norm.size > 0 else 0.0

        return {
            "total": icm_losses["total"],
            "icm_forward": icm_losses["forward"],
            "icm_inverse": icm_losses["inverse"],
            "intrinsic_mean": torch.as_tensor(r_mean, dtype=torch.float32, device=self.device),
        }

    def update(self, obs: Any, next_obs: Any, actions: Any, steps: int = 1) -> dict[str, float]:
        """Train encoder/heads via ICM; report ICM losses + current normalized LP mean."""
        # Measure current normalized LP mean (non-mutating)
        with torch.no_grad():
            o = as_tensor(obs, self.device)
            op = as_tensor(next_obs, self.device)
            a = as_tensor(actions, self.device)
            err, phi_t = self._forward_error_per_sample(o, op, a)
            rids = [self.store.locate(p) for p in phi_t.detach().cpu().numpy()]
            vals = []
            for rid in rids:
                st = self._stats.get(int(rid))
                raw_lp = (
                    0.0 if st is None or st.count == 0 else max(0.0, st.ema_long - st.ema_short)
                )
                vals.append(raw_lp)
            lp_arr = np.asarray(vals, dtype=np.float32)
            lp_norm = self._lp_rms.normalize(lp_arr)
            lp_mean = float(np.mean(lp_norm)) if lp_norm.size > 0 else 0.0

        metrics = self.icm.update(obs, next_obs, actions, steps=steps)
        return {
            "loss_total": float(metrics["loss_total"]),
            "loss_forward": float(metrics["loss_forward"]),
            "loss_inverse": float(metrics["loss_inverse"]),
            "intrinsic_mean": float(self.alpha_lp * lp_mean),
        }

    # ----------------------------- diagnostics ----------------------------

    @property
    def lp_rms(self) -> float:
        """Current RMS used to normalize LP values (for logging/diagnostics)."""
        return self._lp_rms.rms

    def _region_records(self, step: int) -> list[dict]:
        """Build a list of JSON-serializable records for current regions."""
        recs: list[dict] = []
        for leaf in self.store.iter_leaves():
            rid = int(leaf.region_id) if leaf.region_id is not None else -1
            st = self._stats.get(rid)
            ema_l = None if st is None else float(st.ema_long)
            ema_s = None if st is None else float(st.ema_short)
            lp = 0.0
            if st is not None and st.count > 0:
                lp = max(0.0, float(st.ema_long - st.ema_short))
            recs.append(
                {
                    "step": int(step),
                    "region_id": rid,
                    "depth": int(leaf.depth),
                    "count_leaf": int(leaf.count),
                    "ema_long": ema_l,
                    "ema_short": ema_s,
                    "lp": float(lp),
                    "gate": 1,  # Sprint 4 will implement gating; all enabled for now
                    "bbox_lo": (
                        None if leaf.bbox_lo is None else [float(x) for x in leaf.bbox_lo.tolist()]
                    ),
                    "bbox_hi": (
                        None if leaf.bbox_hi is None else [float(x) for x in leaf.bbox_hi.tolist()]
                    ),
                }
            )
        return recs

    def export_diagnostics(self, out_dir: Path, step: int) -> None:
        """Append region stats to `regions.jsonl` and gate states to `gates.csv`.

        Files:
            - regions.jsonl: one JSON object per region per call (includes LP, EMAs, bbox, depth).
            - gates.csv:     rows of "step,region_id,gate" (gate=1 for all, until gating exists).

        Notes:
            * This function is I/O-friendly: creates directories/files if needed and appends lines.
            * Call at a modest cadence (e.g., CSV logging interval) to avoid large files.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        regions_path = out_dir / "regions.jsonl"
        gates_path = out_dir / "gates.csv"

        recs = self._region_records(step=int(step))

        # JSONL append
        with regions_path.open("a", encoding="utf-8") as f_jsonl:
            for r in recs:
                json.dump(r, f_jsonl, ensure_ascii=False)
                f_jsonl.write("\n")

        # CSV append (header once)
        write_header = not gates_path.exists() or gates_path.stat().st_size == 0
        with gates_path.open("a", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            if write_header:
                writer.writerow(["step", "region_id", "gate"])
            for r in recs:
                writer.writerow([int(r["step"]), int(r["region_id"]), int(r["gate"])])
