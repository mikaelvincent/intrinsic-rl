"""Proposed unified intrinsic: α_impact·impact + α_LP·LP with region gating.

- Impact: ||φ(s') − φ(s)||₂ (RIDE-like)
- LP: region-wise learning progress from long/short EMAs (R-IAC-like)
- Gating: suppress regions with low LP and high stochasticity (hysteresis)
- Per-component RMS normalization; outputs flagged as normalized.

Supports both vector and **image** observations by delegating φ(s) to ICM,
which routes vectors through an MLP and images through a ConvEncoder.

Image dtype contract
--------------------
This module accepts **raw uint8 images** (HWC/NHWC/CHW/NCHW) or **float images already
in [0, 1]**. If floats appear to be in [0, 255], the shared preprocessing pipeline
(`utils.image.preprocess_image`) will defensively scale them to [0, 1]. To avoid
accidentally bypassing scaling, do **not** pre-cast images to float32 unless you also
scale to [0, 1]; instead, pass the original arrays/tensors and let preprocessing handle
layout and normalization.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple, Union, List

import numpy as np
import gymnasium as gym
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irl.utils.torchops import as_tensor
from irl.intrinsic.icm import ICM, ICMConfig
from irl.intrinsic.regions import KDTreeRegionStore
from irl.intrinsic.normalization import RunningRMS
from .gating import _RegionStats, update_region_gate
from .normalize import ComponentRMS


class Proposed(nn.Module):
    """Unified intrinsic with gating; outputs are already normalized."""

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
        gate_min_regions_for_gating: int = 3,  # NEW: regions required before medians/gating engage
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("Proposed supports Box observation spaces (vector states or images).")

        self.device = torch.device(device)

        # Backing ICM (shared encoder φ and forward/inverse heads)
        self.icm = icm if icm is not None else ICM(obs_space, act_space, device=device, cfg=icm_cfg)
        self.encoder = self.icm.encoder
        self.is_discrete = self.icm.is_discrete
        self.obs_dim = int(np.prod(obs_space.shape))
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
        # NEW: minimum number of *populated* regions before medians are considered stable.
        self.min_regions_for_gating = int(gate_min_regions_for_gating)
        self._eps = 1e-8
        self.gating_enabled: bool = True

        # Per-component RMS (impact & LP)
        self._rms = ComponentRMS(
            impact=RunningRMS(beta=0.99, eps=1e-8), lp=RunningRMS(beta=0.99, eps=1e-8)
        )
        self.outputs_normalized: bool = True  # trainer should skip global RMS

        self.to(self.device)

    # ----------------------------- internals -----------------------------

    @torch.no_grad()
    def _impact_per_sample(self, obs: Any, next_obs: Any) -> Tensor:
        phi_t = self.icm._phi(obs)
        phi_tp1 = self.icm._phi(next_obs)
        return torch.norm(phi_tp1 - phi_t, p=2, dim=-1)

    @torch.no_grad()
    def _forward_error_and_phi(
        self, obs: Any, next_obs: Any, actions: Any
    ) -> Tuple[Tensor, Tensor]:
        o = obs
        op = next_obs
        a = as_tensor(actions, self.device)
        phi_t = self.icm._phi(o)
        phi_tp1 = self.icm._phi(op)
        a_fwd = self.icm._act_for_forward(a)
        pred = self.icm.forward_head(torch.cat([phi_t, a_fwd], dim=-1))
        per_dim = F.mse_loss(pred, phi_tp1, reduction="none")  # [B, D]
        return per_dim.mean(dim=-1), phi_t

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
        """Update region gate based on LP/stochasticity; return 0/1.

        Medians used by the gating rule only engage once at least
        `self.min_regions_for_gating` regions have observed samples. Until
        then, gating remains permissive (all regions 'on').
        """
        st = self._stats.get(rid)
        if st is None:
            return 1

        # Need ≥ min_regions_for_gating populated regions for robust medians
        sufficient = sum(1 for s in self._stats.values() if s.count > 0) >= int(
            self.min_regions_for_gating
        )
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
        """Single-transition intrinsic with gating and per-component RMS."""
        from irl.intrinsic import IntrinsicOutput  # local import to avoid cycle

        with torch.no_grad():
            s = tr.s
            sp = tr.s_next
            a = tr.a

            impact_raw = self._impact_per_sample(s, sp).view(-1)[0]
            err, phi_t = self._forward_error_and_phi(s, sp, a)
            rid = int(self.store.insert(phi_t.detach().cpu().numpy().reshape(-1)))
            lp_raw = self._update_region(rid, float(err.view(-1)[0].item()))
            gate = self._maybe_update_gate(rid, float(lp_raw)) if self.gating_enabled else 1

            # Normalize components and combine
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
        """Vectorized intrinsic with gating and per-component RMS.

        Optimization: group samples by region id and update region EMAs once per
        region using that region's mean error. Gates are then refreshed per region.
        Per-sample LP/gate inherit their region's post-update values.
        """
        o = obs
        op = next_obs
        a = actions

        impact_raw = self._impact_per_sample(o, op)  # [B]
        err, phi_t = self._forward_error_and_phi(o, op, a)  # [B], [B, D]

        # Map each sample to a region
        phi_np = phi_t.detach().cpu().numpy()
        rids = self.store.bulk_insert(phi_np)  # (B,)
        err_np = err.detach().cpu().numpy().astype(np.float64)

        # Group errors by region id -> mean error per region
        uniq, inv = np.unique(rids, return_inverse=True)
        sums = np.zeros(len(uniq), dtype=np.float64)
        cnts = np.zeros(len(uniq), dtype=np.int64)
        np.add.at(sums, inv, err_np)
        np.add.at(cnts, inv, 1)
        means = sums / np.maximum(1, cnts)

        # 1) Update EMAs once per region
        lp_per_region = np.zeros(len(uniq), dtype=np.float32)
        for i, rid in enumerate(uniq):
            rid_i = int(rid)
            e_mean = float(means[i])
            st = self._stats.get(rid_i)
            if st is None or st.count == 0:
                self._stats[rid_i] = _RegionStats(
                    ema_long=e_mean, ema_short=e_mean, count=int(cnts[i])
                )
                lp = 0.0
            else:
                st.ema_long = self.beta_long * st.ema_long + (1.0 - self.beta_long) * e_mean
                st.ema_short = self.beta_short * st.ema_short + (1.0 - self.beta_short) * e_mean
                st.count += int(cnts[i])
                lp = max(0.0, float(st.ema_long - st.ema_short))
            lp_per_region[i] = float(lp)

        # 2) Refresh gating per region (after EMAs are updated) and broadcast
        gate_per_region = np.ones(len(uniq), dtype=np.int64)
        if self.gating_enabled:
            for i, rid in enumerate(uniq):
                gate_per_region[i] = int(self._maybe_update_gate(int(rid), float(lp_per_region[i])))

        # Broadcast region stats back to samples
        lp_arr = lp_per_region[inv].astype(np.float32)  # (B,)
        gates = gate_per_region[inv].astype(np.float32)  # (B,)

        # Update RMS/normalize per component on a per-sample basis
        imp_np = impact_raw.detach().cpu().numpy().astype(np.float32)
        self._rms.update(imp_np, lp_arr)
        imp_norm, lp_norm = self._rms.normalize(imp_np, lp_arr)

        imp_t = torch.as_tensor(imp_norm, dtype=torch.float32, device=self.device)
        lp_t = torch.as_tensor(lp_norm, dtype=torch.float32, device=self.device)
        gate_t = torch.as_tensor(gates, dtype=torch.float32, device=self.device)

        out = gate_t * (self.alpha_impact * imp_t + self.alpha_lp * lp_t)
        return out.mean() if reduction == "mean" else out

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
            o = obs
            op = next_obs
            a = actions

            impact_raw = self._impact_per_sample(o, op)
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
        """Train ICM and report losses + current normalized intrinsic mean."""
        with torch.no_grad():
            o = obs
            op = next_obs
            a = actions

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
