"""Proposed unified intrinsic module (core implementation).

Combines:
  • RIDE-style impact  : ||φ(s') − φ(s)||₂
  • R-IAC learning progress per region i: LP_i = max(0, EMA_long_i − EMA_short_i)
  • Region-wise gating (randomness filter with hysteresis)

Outputs are normalized per-component (impact & LP) via running RMS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple, Union, List

import numpy as np
import gymnasium as gym
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irl.utils.torchops import as_tensor, ensure_2d
from irl.intrinsic.icm import ICM, ICMConfig
from irl.intrinsic.regions import KDTreeRegionStore
from irl.intrinsic.normalization import RunningRMS
from .gating import _RegionStats, update_region_gate
from .normalize import ComponentRMS


class Proposed(nn.Module):
    """Unified intrinsic = gate_i · (α_impact·norm(impact) + α_LP·norm(LP(region)))."""

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
        # gating knobs
        gate_tau_lp_mult: float = 0.01,
        gate_tau_s: float = 2.0,
        gate_hysteresis_up_mult: float = 2.0,
        gate_min_consec_to_gate: int = 5,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("Proposed supports Box observation spaces (vector states).")

        self.device = torch.device(device)

        # Backing ICM (shared encoder φ and forward/inverse heads)
        self.icm = icm if icm is not None else ICM(obs_space, act_space, device=device, cfg=icm_cfg)
        self.encoder = self.icm.encoder
        self.is_discrete = self.icm.is_discrete
        self.obs_dim = int(obs_space.shape[0])
        self.phi_dim = int(self.icm.cfg.phi_dim)

        # KD-tree regionization over φ-space
        self.store = KDTreeRegionStore(
            dim=self.phi_dim, capacity=int(region_capacity), depth_max=int(depth_max)
        )
        self._stats: dict[int, _RegionStats] = {}

        # Coefficients & EMA knobs
        self.alpha_impact = float(alpha_impact)
        self.alpha_lp = float(alpha_lp)
        self.beta_long = float(ema_beta_long)
        self.beta_short = float(ema_beta_short)

        # Gating thresholds
        self.tau_lp_mult = float(gate_tau_lp_mult)
        self.tau_s = float(gate_tau_s)
        self.hysteresis_up_mult = float(gate_hysteresis_up_mult)
        self.min_consec_to_gate = int(gate_min_consec_to_gate)
        self._eps = 1e-8
        self.gating_enabled: bool = True

        # Per-component RMS (impact & LP)
        self._rms = ComponentRMS(
            impact=RunningRMS(beta=0.99, eps=1e-8), lp=RunningRMS(beta=0.99, eps=1e-8)
        )
        # Signal to trainer that outputs are already normalized here
        self.outputs_normalized: bool = True

        self.to(self.device)

    # ----------------------------- internals -----------------------------

    @torch.no_grad()
    def _impact_per_sample(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        o = ensure_2d(obs)
        op = ensure_2d(next_obs)
        phi_t = self.icm._phi(o)
        phi_tp1 = self.icm._phi(op)
        return torch.norm(phi_tp1 - phi_t, p=2, dim=-1)

    @torch.no_grad()
    def _forward_error_and_phi(
        self, obs: Tensor, next_obs: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor]:
        o = ensure_2d(obs)
        op = ensure_2d(next_obs)
        a = actions
        phi_t = self.icm._phi(o)
        phi_tp1 = self.icm._phi(op)
        a_fwd = self.icm._act_for_forward(a)
        pred = self.icm.forward_head(torch.cat([phi_t, a_fwd], dim=-1))
        per_dim = F.mse_loss(pred, phi_tp1, reduction="none")  # [B, D]
        return per_dim.mean(dim=-1), phi_t  # [B], [B, D]

    def _update_region(self, rid: int, error: float) -> float:
        st = self._stats.get(rid)
        if st is None or st.count == 0:
            self._stats[rid] = _RegionStats(ema_long=error, ema_short=error, count=1)
            return 0.0
        st.ema_long = self.beta_long * st.ema_long + (1.0 - self.beta_long) * error
        st.ema_short = self.beta_short * st.ema_short + (1.0 - self.beta_short) * error
        st.count += 1
        return max(0.0, st.ema_long - st.ema_short)

    def _global_medians(self) -> Tuple[float, float]:
        lps: List[float] = []
        errs: List[float] = []
        for st in self._stats.values():
            if st.count > 0:
                lps.append(max(0.0, float(st.ema_long - st.ema_short)))
                errs.append(float(st.ema_short))
        if len(errs) == 0:
            return 0.0, 0.0
        return float(np.median(lps) if len(lps) > 0 else 0.0), float(np.median(errs))

    # ----------------------------- gating -----------------------------

    def _maybe_update_gate(self, rid: int, lp_i: float) -> int:
        """Update region gate based on LP and stochasticity S_i; return gate."""
        st = self._stats.get(rid)
        if st is None:
            return 1

        # Require a minimum number of regions for robust medians (else keep open)
        sufficient = sum(1 for s in self._stats.values() if s.count > 0) >= 3
        if not sufficient:
            st.bad_consec = 0
            st.good_consec = 0
            st.gate = 1
            return st.gate

        med_lp, med_err = self._global_medians()
        tau_lp = self.tau_lp_mult * med_lp

        return update_region_gate(
            st,
            lp_i=float(lp_i),
            tau_lp=float(tau_lp),
            tau_s=float(self.tau_s),
            median_error_global=float(med_err),
            hysteresis_up_mult=float(self.hysteresis_up_mult),
            min_consec_to_gate=int(self.min_consec_to_gate),
            eps=float(self._eps),
            sufficient_regions=True,
        )

    # ------------------------- public API: compute -------------------------

    def compute(self, tr) -> "IntrinsicOutput":
        """Single-transition intrinsic (gated, per-component normalized)."""
        from irl.intrinsic import IntrinsicOutput  # local import to avoid cycle

        with torch.no_grad():
            s = as_tensor(tr.s, self.device)
            sp = as_tensor(tr.s_next, self.device)
            a = as_tensor(tr.a, self.device)

            impact_raw = self._impact_per_sample(s.view(1, -1), sp.view(1, -1)).view(-1)[0]
            err, phi_t = self._forward_error_and_phi(
                s.view(1, -1), sp.view(1, -1), a.view(1, -1 if not self.is_discrete else 1)
            )
            rid = int(self.store.insert(phi_t.detach().cpu().numpy().reshape(-1)))
            lp_raw = self._update_region(rid, float(err.item()))
            gate = self._maybe_update_gate(rid, float(lp_raw)) if self.gating_enabled else 1

            # Normalize per component
            self._rms.update([float(impact_raw.item())], [float(lp_raw)])
            impact_n, lp_n = self._rms.normalize(
                np.asarray([float(impact_raw.item())], dtype=np.float32),
                np.asarray([float(lp_raw)], dtype=np.float32),
            )
            r = float(gate) * (
                self.alpha_impact * float(impact_n[0]) + self.alpha_lp * float(lp_n[0])
            )
            return IntrinsicOutput(r_int=float(r))

    @torch.no_grad()
    def compute_batch(
        self,
        obs: Any,
        next_obs: Any,
        actions: Any,
        reduction: str = "none",
    ) -> Tensor:
        """Vectorized intrinsic (updates EMAs/gates and **normalizes** components)."""
        o = as_tensor(obs, self.device)
        op = as_tensor(next_obs, self.device)
        a = as_tensor(actions, self.device)

        impact_raw = self._impact_per_sample(o, op)  # [B]
        err, phi_t = self._forward_error_and_phi(o, op, a)  # [B], [B, D]

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

        imp_np = impact_raw.detach().cpu().numpy().astype(np.float32)
        lp_np = np.asarray(lp_vals, dtype=np.float32)

        # update RMS and normalize
        self._rms.update(imp_np, lp_np)
        imp_norm, lp_norm = self._rms.normalize(imp_np, lp_np)

        imp_t = torch.as_tensor(imp_norm, dtype=torch.float32, device=self.device)
        lp_t = torch.as_tensor(lp_norm, dtype=torch.float32, device=self.device)
        gate_t = torch.as_tensor(gates, dtype=torch.float32, device=self.device)

        out = gate_t * (self.alpha_impact * imp_t + self.alpha_lp * lp_t)
        if reduction == "mean":
            return out.mean()
        return out

    # ----------------------------- losses & update -----------------------------

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
        """ICM training loss + diagnostic r_int mean (non-mutating)."""
        icm_losses = self.icm.loss(obs, next_obs, actions)

        with torch.no_grad():
            o = as_tensor(obs, self.device)
            op = as_tensor(next_obs, self.device)
            a = as_tensor(actions, self.device)

            impact_raw = self._impact_per_sample(o, op)  # [B]
            err, phi_t = self._forward_error_and_phi(o, op, a)

            rids = [self.store.locate(p) for p in phi_t.detach().cpu().numpy()]
            lp_now = np.asarray(self._current_lp_for_rids(rids), dtype=np.float32)
            gates_now = torch.as_tensor(
                self._current_gate_for_rids(rids), dtype=torch.float32, device=self.device
            )

            imp_norm, lp_norm = self._rms.normalize(
                impact_raw.detach().cpu().numpy().astype(np.float32), lp_now
            )
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
        """Train ICM and report losses + current normalized intrinsic mean (non-mutating)."""
        with torch.no_grad():
            o = as_tensor(obs, self.device)
            op = as_tensor(next_obs, self.device)
            a = as_tensor(actions, self.device)

            impact_raw = self._impact_per_sample(o, op)
            err, phi_t = self._forward_error_and_phi(o, op, a)
            rids = [self.store.locate(p) for p in phi_t.detach().cpu().numpy()]
            lp_now = np.asarray(self._current_lp_for_rids(rids), dtype=np.float32)

            imp_norm, lp_norm = self._rms.normalize(
                impact_raw.detach().cpu().numpy().astype(np.float32), lp_now
            )
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

    # ----------------------------- diagnostics -----------------------------

    @property
    def gate_rate(self) -> float:
        """Fraction of regions currently gated-off."""
        n = sum(1 for s in self._stats.values() if s.count > 0)
        if n == 0:
            return 0.0
        off = sum(1 for s in self._stats.values() if s.count > 0 and s.gate == 0)
        return float(off) / float(n)

    @property
    def impact_rms(self) -> float:
        """Current RMS for impact component normalization."""
        return float(self._rms.impact.rms)

    @property
    def lp_rms(self) -> float:
        """Current RMS for LP component normalization."""
        return float(self._rms.lp.rms)
