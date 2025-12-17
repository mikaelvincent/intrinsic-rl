from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irl.intrinsic import BaseIntrinsicModule, IntrinsicOutput, Transition
from irl.intrinsic.icm import ICM, ICMConfig
from irl.intrinsic.normalization import RunningRMS
from irl.intrinsic.regions import KDTreeRegionStore
from irl.utils.torchops import as_tensor
from . import diagnostics as _diag


@dataclass
class _RegionStats:
    ema_long: float = 0.0
    ema_short: float = 0.0
    count: int = 0


def simulate_lp_emas(
    errors: Iterable[float], beta_long: float = 0.995, beta_short: float = 0.90
) -> Tuple[list[float], list[float], list[float]]:
    el: list[float] = []
    es: list[float] = []
    lp: list[float] = []
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


class RIAC(BaseIntrinsicModule, nn.Module):
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
        checkpoint_include_points: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("RIAC supports Box observation spaces (vector or image).")
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

        self.beta_long = float(ema_beta_long)
        self.beta_short = float(ema_beta_short)
        self.alpha_lp = float(alpha_lp)

        self._lp_rms = RunningRMS(beta=0.99, eps=1e-8)
        self.outputs_normalized = True
        self.checkpoint_include_points = bool(checkpoint_include_points)
        self.to(self.device)

    def get_extra_state(self) -> dict:
        store_state = self.store.state_dict(include_points=bool(self.checkpoint_include_points))
        if isinstance(store_state, dict):
            store_state = dict(store_state)
            store_state["include_points"] = bool(self.checkpoint_include_points)

        return {
            "store": store_state,
            "stats": {int(k): asdict(v) for k, v in self._stats.items()},
            "rms": self._lp_rms.state_dict(),
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
                self._lp_rms.load_state_dict(rms_state)
        except Exception:
            pass

    @torch.no_grad()
    def _forward_error_per_sample(
        self, obs: Any, next_obs: Any, actions: Any
    ) -> Tuple[Tensor, Tensor]:
        a = as_tensor(actions, self.device)
        phi_t = self.icm._phi(obs)
        phi_tp1 = self.icm._phi(next_obs)
        a_fwd = self.icm._act_for_forward(a)
        pred = self.icm.forward_head(torch.cat([phi_t, a_fwd], dim=-1))
        per_dim = F.mse_loss(pred, phi_tp1, reduction="none")
        return per_dim.mean(dim=-1), phi_t

    def compute(self, tr: Transition) -> IntrinsicOutput:
        with torch.no_grad():
            err, phi_t = self._forward_error_per_sample(tr.s, tr.s_next, tr.a)
            rid = int(self.store.insert(phi_t.detach().cpu().numpy().reshape(-1)))
            lp = self._update_region(rid=int(rid), error=float(err.view(-1)[0].item()))
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
        err_t, phi_t = self._forward_error_per_sample(obs, next_obs, actions)
        phi_np = phi_t.detach().cpu().numpy()
        err_np = err_t.detach().cpu().numpy().astype(np.float64)

        N = int(err_np.shape[0])
        out = np.empty(N, dtype=np.float32)

        rids = self.store.bulk_insert(phi_np)

        for i in range(N):
            rid = int(rids[i])
            lp_val = float(self._update_region(rid, float(err_np[i])))
            self._lp_rms.update([lp_val])
            lp_norm = self._lp_rms.normalize(np.asarray([lp_val], dtype=np.float32))
            out[i] = self.alpha_lp * float(lp_norm[0])

        out_t = torch.as_tensor(out, dtype=torch.float32, device=self.device)
        return out_t.mean() if reduction == "mean" else out_t

    def loss(self, obs: Any, next_obs: Any, actions: Any) -> dict[str, Tensor]:
        icm_losses = self.icm.loss(obs, next_obs, actions)
        with torch.no_grad():
            err, phi_t = self._forward_error_per_sample(obs, next_obs, actions)
            rids = [self.store.locate(p) for p in phi_t.detach().cpu().numpy()]
            lp_vals = []
            for rid in rids:
                st = self._stats.get(int(rid))
                lp_vals.append(
                    0.0 if st is None or st.count == 0 else max(0.0, st.ema_long - st.ema_short)
                )
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
        with torch.no_grad():
            err, phi_t = self._forward_error_per_sample(obs, next_obs, actions)
            rids = [self.store.locate(p) for p in phi_t.detach().cpu().numpy()]
            vals = []
            for rid in rids:
                st = self._stats.get(int(rid))
                vals.append(
                    0.0 if st is None or st.count == 0 else max(0.0, st.ema_long - st.ema_short)
                )
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

    def _update_region(self, rid: int, error: float) -> float:
        st = self._stats.get(rid)
        if st is None or st.count == 0:
            self._stats[rid] = _RegionStats(ema_long=error, ema_short=error, count=1)
            return 0.0
        st.ema_long = self.beta_long * st.ema_long + (1.0 - self.beta_long) * error
        st.ema_short = self.beta_short * st.ema_short + (1.0 - self.beta_short) * error
        st.count += 1
        return max(0.0, st.ema_long - st.ema_short)

    @property
    def lp_rms(self) -> float:
        return self._lp_rms.rms

    def export_diagnostics(self, out_dir: Path, step: int) -> None:
        _diag.export_diagnostics(self.store, self._stats, out_dir=out_dir, step=int(step))
