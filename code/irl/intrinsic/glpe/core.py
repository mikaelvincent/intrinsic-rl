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

        self.to(self.device)

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
                    # If points were omitted, further splits would be history-dependent.
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
        st = self._stats.get(rid)
        if st is None or st.count == 0:
            self._stats[rid] = _RegionStats(ema_long=error, ema_short=error, count=1)
            return 0.0
        st.ema_long = self.beta_long * st.ema_long + (1.0 - self.beta_long) * error
        st.ema_short = self.beta_short * st.ema_short + (1.0 - self.beta_short) * error
        st.count += 1
        return max(0.0, st.ema_long - st.ema_short)

    def _global_medians(self) -> Tuple[float, float]:
        lps: list[float] = []
        errs: list[float] = []
        for st in self._stats.values():
            if st.count > 0:
                lps.append(max(0.0, float(st.ema_long - st.ema_short)))
                errs.append(float(st.ema_short))
        if not errs:
            return 0.0, 0.0
        return float(np.median(lps) if lps else 0.0), float(np.median(errs))

    def _maybe_update_gate(self, rid: int, lp_i: float) -> int:
        st = self._stats.get(rid)
        if st is None:
            return 1

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
                self._rms.update([impact_raw], [lp_raw])
                impact_n, lp_n = self._rms.normalize(
                    np.asarray([impact_raw], dtype=np.float32),
                    np.asarray([lp_raw], dtype=np.float32),
                )
                r = float(gate) * (
                    self.alpha_impact * float(impact_n[0]) + self.alpha_lp * float(lp_n[0])
                )
            else:
                r = float(gate) * (self.alpha_impact * impact_raw + self.alpha_lp * lp_raw)

            return IntrinsicOutput(r_int=float(r))

    @torch.no_grad()
    def compute_batch(self, obs: Any, next_obs: Any, actions: Any, reduction: str = "none") -> Tensor:
        impact_raw = self._impact_per_sample(obs, next_obs)
        err, phi_t = self._forward_error_and_phi(obs, next_obs, actions)

        phi_np = phi_t.detach().cpu().numpy()
        err_np = err.detach().cpu().numpy().astype(np.float64)
        imp_np = impact_raw.detach().cpu().numpy().astype(np.float32)

        N = int(imp_np.shape[0])
        out = np.empty(N, dtype=np.float32)

        rids = self.store.bulk_insert(phi_np)

        for i in range(N):
            rid = int(rids[i])
            lp_raw = float(self._update_region(rid, float(err_np[i])))

            gate = 1
            if self.gating_enabled:
                gate = int(self._maybe_update_gate(rid, float(lp_raw)))

            if self._normalize_inside:
                self._rms.update([float(imp_np[i])], [lp_raw])
                imp_norm, lp_norm = self._rms.normalize(
                    np.asarray([imp_np[i]], dtype=np.float32),
                    np.asarray([lp_raw], dtype=np.float32),
                )
                combined = self.alpha_impact * float(imp_norm[0]) + self.alpha_lp * float(lp_norm[0])
            else:
                combined = self.alpha_impact * float(imp_np[i]) + self.alpha_lp * float(lp_raw)

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
