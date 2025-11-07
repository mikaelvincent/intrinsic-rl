"""Proposed unified intrinsic module (Step 2: RIDE + R‑IAC + gating).

This module combines:
  • RIDE‑style *impact*:   ‖φ(s') − φ(s)‖₂
  • R‑IAC *learning progress* per region i: LP_i = max(0, EMA_long_i − EMA_short_i)
  • **NEW (Sprint 4 — Step 2): Region‑wise gating** based on (§5.4.1):
        Improvement      I_i = LP_i / (ε + EMA_long_i)        [diagnostic]
        Stochasticity    S_i = EMA_short_i / (ε + median_error_global)

    Gate OFF (gate_i = 0) if, for K consecutive refreshes:
        (LP_i < τ_LP) AND (S_i > τ_S)
    Hysteresis: once gated, require LP_i > (hysteresis_up_mult · τ_LP) for 2
    consecutive refreshes to re‑enable the gate (gate_i = 1).

Returned (unnormalized) intrinsic per transition (with gating):
    r_int = gate_i * (α_impact * r_impact + α_LP * LP_i)

Notes
-----
* Outputs were **unnormalized** in Step 2.
* **Sprint 4 — Step 4 (this change):** we now maintain **per‑component running RMS**
  (impact and LP) and return **normalized** outputs:
      r_int = gate_i * (α_impact * norm(impact) + α_LP * norm(LP_i))
  Trainers should **skip** global intrinsic normalization for this module. We
  advertise `outputs_normalized=True`. Final clipping to `[-r_clip, r_clip]`
  remains at the trainer level (already implemented in `irl.train`).
* KD‑tree regionization and EMA updates are performed online, just like Step 1.
* Gating statistics are stored per‑region and updated whenever a region receives
  a new sample (i.e., when LP_i is refreshed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, Iterable, List

import numpy as np
import gymnasium as gym
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import BaseIntrinsicModule, IntrinsicOutput, Transition
from .icm import ICM, ICMConfig
from .regions import KDTreeRegionStore
from .normalization import RunningRMS  # <-- NEW: per-component RMS normalizers
from irl.utils.torchops import as_tensor, ensure_2d


# ------------------------------ Small helpers ------------------------------


@dataclass
class _RegionStats:
    """Per-region EMA container (unnormalized for Step 2/4 gating)."""

    ema_long: float = 0.0
    ema_short: float = 0.0
    count: int = 0
    # ---- gating state (NEW) ----
    gate: int = 1  # 1 => enabled, 0 => gated-off
    bad_consec: int = 0  # consecutive 'bad' refreshes toward gating off
    good_consec: int = 0  # consecutive 'good' refreshes toward re-enabling


# -------------------------------- Proposed ---------------------------------


class Proposed(BaseIntrinsicModule, nn.Module):
    """Unified intrinsic = gate_i · (α_impact·impact + α_LP·LP(region)) with gating.

    Args:
        obs_space: Box observation space (vector states).
        act_space: Discrete or Box action space (for ICM inverse/forward).
        device: torch device string or device.
        icm: Optional external ICM to share encoder/heads; if None, create one.
        icm_cfg: Optional ICMConfig used when constructing an internal ICM.
        alpha_impact: Multiplier for RIDE‑style impact term.
        alpha_lp: Multiplier for region LP term.
        region_capacity: KD-tree leaf capacity before split.
        depth_max: Maximum KD-tree depth.
        ema_beta_long: EMA coefficient for long-term error (R‑IAC).
        ema_beta_short: EMA coefficient for short-term error (R‑IAC).

        # ---- gating knobs ----
        gate_tau_lp_mult: float
            τ_LP = gate_tau_lp_mult * median_LP_global
        gate_tau_s: float
            S_i threshold (S_i > τ_S considered "stochastic").
        gate_hysteresis_up_mult: float
            Require LP_i > (gate_hysteresis_up_mult · τ_LP) to re-enable (for 2 consecutive refreshes).
        gate_min_consec_to_gate: int
            K consecutive "bad" refreshes required to disable (gate_i ← 0).

    Outputs are **normalized** here (Sprint 4 — Step 4) via per-component RMS.
    Trainers should skip global intrinsic normalization for this module.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        device: Union[str, torch.device] = "cpu",
        icm: Optional[ICM] = None,
        icm_cfg: Optional[ICMConfig] = None,
        *,
        alpha_impact: float = 1.0,
        alpha_lp: float = 0.5,
        region_capacity: int = 200,
        depth_max: int = 12,
        ema_beta_long: float = 0.995,
        ema_beta_short: float = 0.90,
        # ---- gating (NEW) ----
        gate_tau_lp_mult: float = 0.01,
        gate_tau_s: float = 2.0,
        gate_hysteresis_up_mult: float = 2.0,
        gate_min_consec_to_gate: int = 5,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("Proposed currently supports Box observation spaces (vector states).")
        self.device = torch.device(device)

        # Backing ICM (shared representation φ, forward/inverse heads)
        self.icm = icm if icm is not None else ICM(obs_space, act_space, device=device, cfg=icm_cfg)
        self.encoder = self.icm.encoder  # explicit alias
        self.is_discrete = self.icm.is_discrete
        self.obs_dim = int(obs_space.shape[0])
        self.phi_dim = int(self.icm.cfg.phi_dim)

        # R‑IAC regionization in φ-space
        self.store = KDTreeRegionStore(
            dim=self.phi_dim, capacity=int(region_capacity), depth_max=int(depth_max)
        )
        self._stats: dict[int, _RegionStats] = {}

        # Coefficients / EMA knobs
        self.alpha_impact = float(alpha_impact)
        self.alpha_lp = float(alpha_lp)
        self.beta_long = float(ema_beta_long)
        self.beta_short = float(ema_beta_short)

        # ---- gating knobs (NEW) ----
        self.tau_lp_mult = float(gate_tau_lp_mult)
        self.tau_s = float(gate_tau_s)
        self.hysteresis_up_mult = float(gate_hysteresis_up_mult)
        self.min_consec_to_gate = int(gate_min_consec_to_gate)
        self._eps = 1e-8
        self.gating_enabled: bool = True

        # ---- NEW (Sprint 4 — Step 4): per-component RMS normalizers ----
        self._impact_rms = RunningRMS(beta=0.99, eps=1e-8)
        self._lp_rms = RunningRMS(beta=0.99, eps=1e-8)

        # Outputs are **pre-normalized** inside this module
        self.outputs_normalized: bool = True

        self.to(self.device)

    # -------------------------- Core components --------------------------

    @torch.no_grad()
    def _impact_per_sample(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        """RIDE impact magnitude: ||φ(s') - φ(s)||₂, shape [B]."""
        o = ensure_2d(obs)
        op = ensure_2d(next_obs)
        phi_t = self.icm._phi(o)
        phi_tp1 = self.icm._phi(op)
        return torch.norm(phi_tp1 - phi_t, p=2, dim=-1)

    @torch.no_grad()
    def _forward_error_and_phi(
        self, obs: Tensor, next_obs: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Return (per-sample forward MSE in φ space [B], phi(s) [B,D])."""
        o = ensure_2d(obs)
        op = ensure_2d(next_obs)
        a = actions
        phi_t = self.icm._phi(o)
        phi_tp1 = self.icm._phi(op)
        a_fwd = self.icm._act_for_forward(a)
        pred = self.icm.forward_head(torch.cat([phi_t, a_fwd], dim=-1))
        per_dim = F.mse_loss(pred, phi_tp1, reduction="none")  # [B,D]
        return per_dim.mean(dim=-1), phi_t  # [B], [B,D]

    def _update_region(self, rid: int, error: float) -> float:
        """Update EMAs for region rid with forward error; return LP (>=0)."""
        st = self._stats.get(rid)
        if st is None or st.count == 0:
            self._stats[rid] = _RegionStats(ema_long=error, ema_short=error, count=1)
            return 0.0  # first observation: define LP=0
        st.ema_long = self.beta_long * st.ema_long + (1.0 - self.beta_long) * error
        st.ema_short = self.beta_short * st.ema_short + (1.0 - self.beta_short) * error
        st.count += 1
        return max(0.0, st.ema_long - st.ema_short)

    # -------- NEW: gating metrics, thresholds and update per region --------

    def _global_medians(self) -> Tuple[float, float]:
        """Return (median_LP_global, median_error_global) over regions with samples."""
        lps: List[float] = []
        errs: List[float] = []
        for st in self._stats.values():
            if st.count > 0:
                lps.append(max(0.0, float(st.ema_long - st.ema_short)))
                errs.append(float(st.ema_short))
        if len(errs) == 0:
            return 0.0, 0.0
        return float(np.median(lps) if len(lps) > 0 else 0.0), float(np.median(errs))

    def _maybe_update_gate(self, rid: int, lp_i: float) -> int:
        """Update region gate state based on LP and stochasticity score S_i.

        Returns the current gate (1 enabled, 0 gated-off).
        """
        st = self._stats.get(rid)
        if st is None:
            return 1

        # Require a minimum number of regions for robust medians (else keep open)
        if sum(1 for s in self._stats.values() if s.count > 0) < 3:
            st.bad_consec = 0
            st.good_consec = 0
            st.gate = 1
            return st.gate

        med_lp, med_err = self._global_medians()
        tau_lp = self.tau_lp_mult * med_lp
        # Stochasticity score S_i = EMA_short / (eps + median_error_global)
        s_i = float(st.ema_short) / (self._eps + float(med_err))

        # "Bad" condition toward gating off
        cond_bad = (lp_i < tau_lp) and (s_i > self.tau_s)

        if st.gate == 1:
            if cond_bad:
                st.bad_consec += 1
                if st.bad_consec >= self.min_consec_to_gate:
                    st.gate = 0
                    st.good_consec = 0
            else:
                st.bad_consec = 0
        else:
            # Currently gated off; check hysteresis to re-enable
            if lp_i > (self.hysteresis_up_mult * tau_lp):
                st.good_consec += 1
                if st.good_consec >= 2:
                    st.gate = 1
                    st.bad_consec = 0
            else:
                st.good_consec = 0

        return st.gate

    # -------------------------- Intrinsic compute -------------------------

    def compute(self, tr: Transition) -> IntrinsicOutput:
        """Compute gated α_imp·norm(impact) + α_LP·norm(LP(region)) for a single transition."""
        with torch.no_grad():
            s = as_tensor(tr.s, self.device)
            sp = as_tensor(tr.s_next, self.device)
            a = as_tensor(tr.a, self.device)

            # Components (raw)
            impact_raw = self._impact_per_sample(s.view(1, -1), sp.view(1, -1)).view(-1)[0]
            err, phi_t = self._forward_error_and_phi(
                s.view(1, -1), sp.view(1, -1), a.view(1, -1 if not self.is_discrete else 1)
            )
            rid = int(self.store.insert(phi_t.detach().cpu().numpy().reshape(-1)))
            lp_raw = self._update_region(rid, float(err.item()))
            gate = self._maybe_update_gate(rid, float(lp_raw)) if self.gating_enabled else 1

            # Normalize per component (update running RMS)
            self._impact_rms.update([float(impact_raw.item())])
            self._lp_rms.update([float(lp_raw)])
            impact_n = float(self._impact_rms.normalize([float(impact_raw.item())])[0])
            lp_n = float(self._lp_rms.normalize([float(lp_raw)])[0])

            r = float(gate) * (self.alpha_impact * impact_n + self.alpha_lp * lp_n)
            return IntrinsicOutput(r_int=float(r))

    @torch.no_grad()
    def compute_batch(
        self,
        obs: Any,
        next_obs: Any,
        actions: Any,
        reduction: str = "none",
    ) -> Tensor:
        """Vectorized intrinsic for a batch (updates RIAC EMAs/gates and **normalizes** components)."""
        o = as_tensor(obs, self.device)
        op = as_tensor(next_obs, self.device)
        a = as_tensor(actions, self.device)

        # Raw components
        impact_raw = self._impact_per_sample(o, op)  # [B]
        err, phi_t = self._forward_error_and_phi(o, op, a)  # [B], [B,D]

        # Region IDs and LP/gate updates
        phi_np = phi_t.detach().cpu().numpy()
        rids = self.store.bulk_insert(phi_np)  # [B]
        lp_vals: list[float] = []
        gates: list[int] = []
        for i in range(impact_raw.shape[0]):
            rid = int(rids[i])
            lp = self._update_region(rid, float(err[i].item()))
            lp_vals.append(lp)
            g = self._maybe_update_gate(rid, float(lp)) if self.gating_enabled else 1
            gates.append(int(g))

        # ---- Normalize per component via running RMS (update on this batch) ----
        # Move to CPU numpy for RunningRMS; return float32 arrays
        imp_np = impact_raw.detach().cpu().numpy().astype(np.float32)
        lp_np = np.asarray(lp_vals, dtype=np.float32)

        self._impact_rms.update(imp_np)
        self._lp_rms.update(lp_np)

        imp_norm = self._impact_rms.normalize(imp_np)  # np.float32 [B]
        lp_norm = self._lp_rms.normalize(lp_np)  # np.float32 [B]

        # Back to torch
        imp_t = torch.as_tensor(imp_norm, dtype=torch.float32, device=self.device)
        lp_t = torch.as_tensor(lp_norm, dtype=torch.float32, device=self.device)
        gate_t = torch.as_tensor(gates, dtype=torch.float32, device=self.device)  # [B]

        out = gate_t * (self.alpha_impact * imp_t + self.alpha_lp * lp_t)  # [B]
        if reduction == "mean":
            return out.mean()
        return out

    # ----------------------------- Loss & Update ---------------------------

    def _current_lp_for_rids(self, rids: Iterable[int]) -> list[float]:
        vals: list[float] = []
        for rid in rids:
            st = self._stats.get(int(rid))
            raw_lp = 0.0 if st is None or st.count == 0 else max(0.0, st.ema_long - st.ema_short)
            vals.append(float(raw_lp))
        return vals

    def _current_gate_for_rids(self, rids: Iterable[int]) -> list[int]:
        g: list[int] = []
        for rid in rids:
            st = self._stats.get(int(rid))
            g.append(1 if (st is None) else int(st.gate))
        return g

    def loss(self, obs: Any, next_obs: Any, actions: Any) -> dict[str, Tensor]:
        """ICM training loss + diagnostic r_int mean (non-mutating; uses current RMS for normalization)."""
        icm_losses = self.icm.loss(obs, next_obs, actions)

        with torch.no_grad():
            o = as_tensor(obs, self.device)
            op = as_tensor(next_obs, self.device)
            a = as_tensor(actions, self.device)

            impact_raw = self._impact_per_sample(o, op)  # [B]
            err, phi_t = self._forward_error_and_phi(o, op, a)  # [B], [B,D]

            # Use current region stats (no mutation) for LP and gate snapshot
            rids = [self.store.locate(p) for p in phi_t.detach().cpu().numpy()]
            lp_now = np.asarray(self._current_lp_for_rids(rids), dtype=np.float32)  # [B]
            gates_now = torch.as_tensor(
                self._current_gate_for_rids(rids), dtype=torch.float32, device=self.device
            )

            # Normalize with *current* RMS (do not update in loss)
            imp_norm = self._impact_rms.normalize(
                impact_raw.detach().cpu().numpy().astype(np.float32)
            )
            lp_norm = self._lp_rms.normalize(lp_now)

            imp_t = torch.as_tensor(imp_norm, dtype=torch.float32, device=self.device)
            lp_t = torch.as_tensor(lp_norm, dtype=torch.float32, device=self.device)

            r_mean = (gates_now * (self.alpha_impact * imp_t + self.alpha_lp * lp_t)).mean()

        return {
            "total": icm_losses["total"],
            "icm_forward": icm_losses["forward"],
            "icm_inverse": icm_losses["inverse"],
            "intrinsic_mean": r_mean,
        }

    def update(self, obs: Any, next_obs: Any, actions: Any, steps: int = 1) -> dict[str, float]:
        """Train ICM (shared encoder/heads) and report losses + r_int mean (non-mutating)."""
        with torch.no_grad():
            o = as_tensor(obs, self.device)
            op = as_tensor(next_obs, self.device)
            a = as_tensor(actions, self.device)

            impact_raw = self._impact_per_sample(o, op)  # [B]
            err, phi_t = self._forward_error_and_phi(o, op, a)
            rids = [self.store.locate(p) for p in phi_t.detach().cpu().numpy()]
            lp_now = np.asarray(self._current_lp_for_rids(rids), dtype=np.float32)

            # Normalize with *current* RMS (do not update here)
            imp_norm = self._impact_rms.normalize(
                impact_raw.detach().cpu().numpy().astype(np.float32)
            )
            lp_norm = self._lp_rms.normalize(lp_now)

            imp_t = torch.as_tensor(imp_norm, dtype=torch.float32, device=self.device)
            lp_t = torch.as_tensor(lp_norm, dtype=torch.float32, device=self.device)
            gates_now = torch.as_tensor(
                self._current_gate_for_rids(rids), dtype=torch.float32, device=self.device
            )
            r_mean = float(
                (gates_now * (self.alpha_impact * imp_t + self.alpha_lp * lp_t)).mean().item()
            )

        metrics = self.icm.update(obs, next_obs, actions, steps=steps)
        return {
            "loss_total": float(metrics["loss_total"]),
            "loss_forward": float(metrics["loss_forward"]),
            "loss_inverse": float(metrics["loss_inverse"]),
            "intrinsic_mean": float(r_mean),
        }

    # ----------------------------- diagnostics ----------------------------

    @property
    def gate_rate(self) -> float:
        """Fraction of regions currently gated-off (for optional logging/diagnostics)."""
        n = sum(1 for s in self._stats.values() if s.count > 0)
        if n == 0:
            return 0.0
        off = sum(1 for s in self._stats.values() if s.count > 0 and s.gate == 0)
        return float(off) / float(n)

    # ---- NEW: expose RMS values for logging parity with RIAC ----
    @property
    def impact_rms(self) -> float:
        """Current RMS for impact component normalization."""
        return float(self._impact_rms.rms)

    @property
    def lp_rms(self) -> float:
        """Current RMS for LP component normalization (name aligns with RIAC)."""
        return float(self._lp_rms.rms)
