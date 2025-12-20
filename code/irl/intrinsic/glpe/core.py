from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irl.intrinsic.icm import ICM, ICMConfig
from irl.intrinsic.normalization import RunningRMS
from irl.intrinsic.regions import KDTreeRegionStore
from irl.utils.torchops import as_tensor

from .gating import _RegionStats, update_region_gate
from .normalize import ComponentRMS


class GLPE(nn.Module):
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
        gate_tau_lp_mult: float = 0.01,
        gate_tau_s: float = 2.0,
        gate_hysteresis_up_mult: float = 2.0,
        gate_min_consec_to_gate: int = 5,
        gate_min_regions_for_gating: int = 3,
        gate_median_cache_interval: int = 1,
        normalize_inside: bool = True,
        gating_enabled: bool = True,
        checkpoint_include_points: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("GLPE supports Box observation spaces (vector states or images).")

        self.device = torch.device(device)

        self.icm = icm if icm is not None else ICM(obs_space, act_space, device=device, cfg=icm_cfg)
        self.encoder = self.icm.encoder
        self.is_discrete = self.icm.is_discrete
        self.obs_dim = int(np.prod(obs_space.shape))
        self.phi_dim = int(self.icm.cfg.phi_dim)

        self.store = KDTreeRegionStore(
            dim=self.phi_dim, capacity=int(region_capacity), depth_max=int(depth_max)
        )
        self._stats: dict[int, _RegionStats] = {}

        self._regions_with_stats: int = 0
        self._stats_ema_long = np.zeros((0,), dtype=np.float64)
        self._stats_ema_short = np.zeros((0,), dtype=np.float64)
        self._stats_count = np.zeros((0,), dtype=np.int64)

        self.alpha_impact = float(alpha_impact)
        self.alpha_lp = float(alpha_lp)
        self.beta_long = float(ema_beta_long)
        self.beta_short = float(ema_beta_short)

        self.tau_lp_mult = float(gate_tau_lp_mult)
        self.tau_s = float(gate_tau_s)
        self.hysteresis_up_mult = float(gate_hysteresis_up_mult)
        self.min_consec_to_gate = int(gate_min_consec_to_gate)
        self.min_regions_for_gating = int(gate_min_regions_for_gating)
        self._eps = 1e-8

        self.gating_enabled = bool(gating_enabled)
        self._normalize_inside = bool(normalize_inside)
        self.checkpoint_include_points = bool(checkpoint_include_points)

        self._rms = ComponentRMS(
            impact=RunningRMS(beta=0.99, eps=1e-8), lp=RunningRMS(beta=0.99, eps=1e-8)
        )
        self.outputs_normalized = bool(self._normalize_inside)

        self._gate_median_cache_interval: int = 1
        self._gate_median_update_count: int = 0
        self._gate_median_cache_last_update: int = -1
        self._gate_median_cache_regions: int = -1
        self._gate_median_cache_lp: float = 0.0
        self._gate_median_cache_err: float = 0.0
        self.gate_median_cache_interval = int(gate_median_cache_interval)

        self.to(self.device)

    @property
    def gate_median_cache_interval(self) -> int:
        return int(self._gate_median_cache_interval)

    @gate_median_cache_interval.setter
    def gate_median_cache_interval(self, v: int) -> None:
        iv = int(v)
        if iv < 1:
            iv = 1
        self._gate_median_cache_interval = iv
        self._reset_gate_median_cache()

    def _invalidate_gate_median_cache(self) -> None:
        self._gate_median_cache_last_update = -1
        self._gate_median_cache_regions = -1

    def _reset_gate_median_cache(self) -> None:
        self._gate_median_update_count = 0
        self._gate_median_cache_last_update = -1
        self._gate_median_cache_regions = -1
        self._gate_median_cache_lp = 0.0
        self._gate_median_cache_err = 0.0

    def _ensure_stats_capacity(self, rid: int) -> None:
        need = int(rid) + 1
        n = int(self._stats_count.size)
        if n >= need:
            return

        new_n = max(need, max(16, n * 2))
        ema_l = np.zeros((new_n,), dtype=np.float64)
        ema_s = np.zeros((new_n,), dtype=np.float64)
        cnt = np.zeros((new_n,), dtype=np.int64)

        if n:
            ema_l[:n] = self._stats_ema_long
            ema_s[:n] = self._stats_ema_short
            cnt[:n] = self._stats_count

        self._stats_ema_long = ema_l
        self._stats_ema_short = ema_s
        self._stats_count = cnt

    def _rebuild_stats_cache(self) -> None:
        if not self._stats:
            self._regions_with_stats = 0
            self._stats_ema_long = np.zeros((0,), dtype=np.float64)
            self._stats_ema_short = np.zeros((0,), dtype=np.float64)
            self._stats_count = np.zeros((0,), dtype=np.int64)
            self._reset_gate_median_cache()
            return

        keys = [
            int(k)
            for k in self._stats.keys()
            if isinstance(k, (int, float)) or str(k).lstrip("-").isdigit()
        ]
        if not keys:
            self._regions_with_stats = 0
            self._stats_ema_long = np.zeros((0,), dtype=np.float64)
            self._stats_ema_short = np.zeros((0,), dtype=np.float64)
            self._stats_count = np.zeros((0,), dtype=np.int64)
            self._reset_gate_median_cache()
            return

        max_rid = max(int(k) for k in keys if int(k) >= 0) if any(int(k) >= 0 for k in keys) else -1
        if max_rid < 0:
            self._regions_with_stats = 0
            self._stats_ema_long = np.zeros((0,), dtype=np.float64)
            self._stats_ema_short = np.zeros((0,), dtype=np.float64)
            self._stats_count = np.zeros((0,), dtype=np.int64)
            self._reset_gate_median_cache()
            return

        n = int(max_rid) + 1
        self._stats_ema_long = np.zeros((n,), dtype=np.float64)
        self._stats_ema_short = np.zeros((n,), dtype=np.float64)
        self._stats_count = np.zeros((n,), dtype=np.int64)

        regions = 0
        for rid, st in self._stats.items():
            try:
                r = int(rid)
            except Exception:
                continue
            if r < 0 or r >= n:
                continue

            c = int(getattr(st, "count", 0))
            self._stats_ema_long[r] = float(getattr(st, "ema_long", 0.0))
            self._stats_ema_short[r] = float(getattr(st, "ema_short", 0.0))
            self._stats_count[r] = c
            if c > 0:
                regions += 1

        self._regions_with_stats = int(regions)
        self._reset_gate_median_cache()

    def get_extra_state(self) -> dict:
        store_state = self.store.state_dict(include_points=bool(self.checkpoint_include_points))
        if isinstance(store_state, dict):
            store_state = dict(store_state)
            store_state["include_points"] = bool(self.checkpoint_include_points)

        return {
            "store": store_state,
            "stats": {int(k): asdict(v) for k, v in self._stats.items()},
            "rms": self._rms.state_dict(),
        }

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return

        try:
            store_state = state.get("store")
            if isinstance(store_state, dict):
                includes_points = bool(store_state.get("include_points", True))
                self.store = KDTreeRegionStore.from_state_dict(store_state)
                if not includes_points:
                    self.store.depth_max = 0
        except Exception:
            pass

        try:
            stats_state = state.get("stats")
            if isinstance(stats_state, dict):
                restored: dict[int, _RegionStats] = {}
                for k, v in stats_state.items():
                    try:
                        rid = int(k)
                    except Exception:
                        continue
                    if not isinstance(v, dict):
                        continue
                    try:
                        restored[rid] = _RegionStats(**v)
                    except Exception:
                        st = _RegionStats()
                        for fk, fv in v.items():
                            if hasattr(st, fk):
                                try:
                                    setattr(st, fk, fv)
                                except Exception:
                                    pass
                        restored[rid] = st
                self._stats = restored
                self._rebuild_stats_cache()
        except Exception:
            pass

        try:
            rms_state = state.get("rms")
            if isinstance(rms_state, dict):
                self._rms.load_state_dict(rms_state)
        except Exception:
            pass

    @torch.no_grad()
    def _impact_per_sample(self, obs: Any, next_obs: Any) -> Tensor:
        phi_t = self.icm._phi(obs)
        phi_tp1 = self.icm._phi(next_obs)
        return torch.norm(phi_tp1 - phi_t, p=2, dim=-1)

    @torch.no_grad()
    def _forward_error_and_phi(self, obs: Any, next_obs: Any, actions: Any) -> Tuple[Tensor, Tensor]:
        a = as_tensor(actions, self.device)
        phi_t = self.icm._phi(obs)
        phi_tp1 = self.icm._phi(next_obs)
        a_fwd = self.icm._act_for_forward(a)
        pred = self.icm.forward_head(torch.cat([phi_t, a_fwd], dim=-1))
        per_dim = F.mse_loss(pred, phi_tp1, reduction="none")
        return per_dim.mean(dim=-1), phi_t

    def _update_region(self, rid: int, error: float) -> float:
        rid_i = int(rid)
        self._ensure_stats_capacity(rid_i)

        invalidate = False
        st = self._stats.get(rid_i)
        if st is None or st.count == 0:
            prev_count = int(self._stats_count[rid_i]) if rid_i < int(self._stats_count.size) else 0
            self._stats[rid_i] = _RegionStats(ema_long=error, ema_short=error, count=1)
            self._stats_ema_long[rid_i] = float(error)
            self._stats_ema_short[rid_i] = float(error)
            self._stats_count[rid_i] = 1
            if prev_count <= 0:
                self._regions_with_stats += 1
                invalidate = True
            else:
                invalidate = True

            self._gate_median_update_count += 1
            if invalidate:
                self._invalidate_gate_median_cache()
            return 0.0

        st.ema_long = self.beta_long * st.ema_long + (1.0 - self.beta_long) * error
        st.ema_short = self.beta_short * st.ema_short + (1.0 - self.beta_short) * error
        st.count += 1

        self._stats_ema_long[rid_i] = float(st.ema_long)
        self._stats_ema_short[rid_i] = float(st.ema_short)
        self._stats_count[rid_i] = int(st.count)

        self._gate_median_update_count += 1

        return max(0.0, st.ema_long - st.ema_short)

    def _global_medians(self) -> Tuple[float, float]:
        if int(self._regions_with_stats) <= 0:
            return 0.0, 0.0

        cnt = self._stats_count
        if cnt.size == 0:
            return 0.0, 0.0

        mask = cnt > 0
        if not bool(np.any(mask)):
            return 0.0, 0.0

        ema_s = self._stats_ema_short[mask]
        ema_l = self._stats_ema_long[mask]
        lp = ema_l - ema_s
        np.maximum(lp, 0.0, out=lp)

        med_lp = float(np.median(lp)) if lp.size > 0 else 0.0
        med_err = float(np.median(ema_s)) if ema_s.size > 0 else 0.0
        return med_lp, med_err

    def _cached_global_medians(self) -> Tuple[float, float]:
        interval = int(self._gate_median_cache_interval)
        if interval <= 1:
            return self._global_medians()

        if (
            self._gate_median_cache_last_update < 0
            or int(self._gate_median_cache_regions) != int(self._regions_with_stats)
        ):
            med_lp, med_err = self._global_medians()
            self._gate_median_cache_lp = float(med_lp)
            self._gate_median_cache_err = float(med_err)
            self._gate_median_cache_last_update = int(self._gate_median_update_count)
            self._gate_median_cache_regions = int(self._regions_with_stats)
            return float(med_lp), float(med_err)

        if (int(self._gate_median_update_count) - int(self._gate_median_cache_last_update)) >= interval:
            med_lp, med_err = self._global_medians()
            self._gate_median_cache_lp = float(med_lp)
            self._gate_median_cache_err = float(med_err)
            self._gate_median_cache_last_update = int(self._gate_median_update_count)
            self._gate_median_cache_regions = int(self._regions_with_stats)

        return float(self._gate_median_cache_lp), float(self._gate_median_cache_err)

    def _maybe_update_gate(self, rid: int, lp_i: float) -> int:
        st = self._stats.get(rid)
        if st is None:
            return 1

        sufficient = int(self._regions_with_stats) >= int(self.min_regions_for_gating)
        if not sufficient:
            st.bad_consec = 0
            st.good_consec = 0
            st.gate = 1
            return st.gate

        # Caching changes gating decisions when interval>1 by using slightly stale global medians.
        med_lp, med_err = self._cached_global_medians()
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

    def compute(self, tr) -> "IntrinsicOutput":
        from irl.intrinsic import IntrinsicOutput

        with torch.no_grad():
            s = tr.s
            sp = tr.s_next
            a = tr.a

            impact_raw_t = self._impact_per_sample(s, sp).view(-1)[0]
            impact_raw = float(impact_raw_t.item())

            err, phi_t = self._forward_error_and_phi(s, sp, a)
            rid = int(self.store.insert(phi_t.detach().cpu().numpy().reshape(-1)))
            lp_raw = float(self._update_region(rid, float(err.view(-1)[0].item())))
            gate = self._maybe_update_gate(rid, float(lp_raw)) if self.gating_enabled else 1

            if self._normalize_inside:
                self._rms.update_scalar(impact_raw, lp_raw)
                impact_n, lp_n = self._rms.normalize_scalar(impact_raw, lp_raw)
                r = float(gate) * (self.alpha_impact * float(impact_n) + self.alpha_lp * float(lp_n))
            else:
                r = float(gate) * (self.alpha_impact * impact_raw + self.alpha_lp * lp_raw)

            return IntrinsicOutput(r_int=float(r))

    @torch.no_grad()
    def compute_batch(self, obs: Any, next_obs: Any, actions: Any, reduction: str = "none") -> Tensor:
        impact_raw = self._impact_per_sample(obs, next_obs)
        err, phi_t = self._forward_error_and_phi(obs, next_obs, actions)

        phi_np = phi_t.detach().cpu().numpy()
        err_np = err.detach().cpu().numpy()
        imp_np = impact_raw.detach().cpu().numpy()

        N = int(imp_np.shape[0])
        out = np.empty(N, dtype=np.float32)

        rids = self.store.bulk_insert(phi_np)

        gating_enabled = bool(self.gating_enabled)
        normalize_inside = bool(self._normalize_inside)
        alpha_imp = float(self.alpha_impact)
        alpha_lp = float(self.alpha_lp)

        update_region = self._update_region
        maybe_update_gate = self._maybe_update_gate
        rms = self._rms

        for i in range(N):
            rid = int(rids[i])
            lp_raw = float(update_region(rid, float(err_np[i])))

            gate = int(maybe_update_gate(rid, float(lp_raw))) if gating_enabled else 1

            if normalize_inside:
                imp_i = float(imp_np[i])
                rms.update_scalar(imp_i, lp_raw)
                imp_norm, lp_norm = rms.normalize_scalar(imp_i, lp_raw)
                combined = alpha_imp * float(imp_norm) + alpha_lp * float(lp_norm)
            else:
                combined = alpha_imp * float(imp_np[i]) + alpha_lp * float(lp_raw)

            out[i] = float(gate) * float(combined)

        out_t = torch.as_tensor(out, dtype=torch.float32, device=self.device)
        return out_t.mean() if reduction == "mean" else out_t

    def _current_lp_for_rids(self, rids: Iterable[int]) -> list[float]:
        vals: list[float] = []
        for rid in rids:
            st = self._stats.get(int(rid))
            raw_lp = 0.0 if st is None or st.count == 0 else max(0.0, st.ema_long - st.ema_short)
            vals.append(float(raw_lp))
        return vals

    def _current_gate_for_rids(self, rids: Iterable[int]) -> list[int]:
        gates: list[int] = []
        for rid in rids:
            st = self._stats.get(int(rid))
            gates.append(1 if st is None else int(st.gate))
        return gates

    def loss(self, obs: Any, next_obs: Any, actions: Any) -> dict[str, Tensor]:
        icm_losses = self.icm.loss(obs, next_obs, actions)

        with torch.no_grad():
            impact_raw = self._impact_per_sample(obs, next_obs)
            _, phi_t = self._forward_error_and_phi(obs, next_obs, actions)

            rids = [self.store.locate(p) for p in phi_t.detach().cpu().numpy()]
            lp_now = np.asarray(self._current_lp_for_rids(rids), dtype=np.float32)
            gates_now = torch.as_tensor(
                self._current_gate_for_rids(rids), dtype=torch.float32, device=self.device
            )

            if self._normalize_inside:
                imp_norm, lp_norm = self._rms.normalize(
                    impact_raw.detach().cpu().numpy().astype(np.float32), lp_now
                )
                imp_t = torch.as_tensor(imp_norm, dtype=torch.float32, device=self.device)
                lp_t = torch.as_tensor(lp_norm, dtype=torch.float32, device=self.device)
            else:
                imp_t = impact_raw.to(dtype=torch.float32, device=self.device)
                lp_t = torch.as_tensor(lp_now, dtype=torch.float32, device=self.device)

            r_mean = (gates_now * (self.alpha_impact * imp_t + self.alpha_lp * lp_t)).mean()

        return {
            "total": icm_losses["total"],
            "icm_forward": icm_losses["forward"],
            "icm_inverse": icm_losses["inverse"],
            "intrinsic_mean": r_mean,
        }

    def update(self, obs: Any, next_obs: Any, actions: Any, steps: int = 1) -> dict[str, float]:
        with torch.no_grad():
            impact_raw = self._impact_per_sample(obs, next_obs)
            _, phi_t = self._forward_error_and_phi(obs, next_obs, actions)
            rids = [self.store.locate(p) for p in phi_t.detach().cpu().numpy()]
            lp_now = np.asarray(self._current_lp_for_rids(rids), dtype=np.float32)
            gates_now = torch.as_tensor(
                self._current_gate_for_rids(rids), dtype=torch.float32, device=self.device
            )

            if self._normalize_inside:
                imp_norm, lp_norm = self._rms.normalize(
                    impact_raw.detach().cpu().numpy().astype(np.float32), lp_now
                )
                imp_t = torch.as_tensor(imp_norm, dtype=torch.float32, device=self.device)
                lp_t = torch.as_tensor(lp_norm, dtype=torch.float32, device=self.device)
            else:
                imp_t = impact_raw.to(dtype=torch.float32, device=self.device)
                lp_t = torch.as_tensor(lp_now, dtype=torch.float32, device=self.device)

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

    @property
    def gate_rate(self) -> float:
        n = sum(1 for s in self._stats.values() if s.count > 0)
        if n == 0:
            return 0.0
        off = sum(1 for s in self._stats.values() if s.count > 0 and s.gate == 0)
        return float(off) / float(n)

    @property
    def impact_rms(self) -> float:
        return float(self._rms.impact.rms)

    @property
    def lp_rms(self) -> float:
        return float(self._rms.lp.rms)
