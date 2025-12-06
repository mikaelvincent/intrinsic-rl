"""R-IAC intrinsic: region-wise learning progress in φ-space.

Maintains long/short EMAs of forward error per region; LP = max(0, EMA_long - EMA_short).
Returns LP normalized via a running RMS and scaled by α_LP.

Supports **image** observations by delegating φ(s) to ICM, which routes
vectors via an MLP and images via a ConvEncoder.

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
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irl.intrinsic import BaseIntrinsicModule, IntrinsicOutput, Transition
from irl.intrinsic.icm import ICM, ICMConfig
from irl.intrinsic.regions import KDTreeRegionStore
from irl.intrinsic.normalization import RunningRMS
from irl.utils.torchops import as_tensor
from . import diagnostics as _diag


@dataclass
class _RegionStats:
    """Per-region EMAs (long/short) and sample count."""

    ema_long: float = 0.0
    ema_short: float = 0.0
    count: int = 0


def simulate_lp_emas(
    errors: Iterable[float], beta_long: float = 0.995, beta_short: float = 0.90
) -> Tuple[list[float], list[float], list[float]]:
    """Return trajectories of EMA_long, EMA_short, and LP for a stream of errors."""
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
    """R-IAC: learning progress per region (normalized)."""

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
            raise TypeError("RIAC supports Box observation spaces (vector or image).")
        self.device = torch.device(device)

        # ICM backbone (shared encoder/forward)
        self.icm = icm if icm is not None else ICM(obs_space, act_space, device=device, cfg=icm_cfg)
        self.encoder = self.icm.encoder
        self.is_discrete = self.icm.is_discrete
        self.obs_dim = int(np.prod(obs_space.shape))
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

        # Running RMS over LP values
        self._lp_rms = RunningRMS(beta=0.99, eps=1e-8)
        self.outputs_normalized: bool = True  # for trainer integration

        self.to(self.device)

    # -------------------------- intrinsic computation -------------------------

    @torch.no_grad()
    def _forward_error_per_sample(
        self, obs: Any, next_obs: Any, actions: Any
    ) -> Tuple[Tensor, Tensor]:
        """Return mean forward error per sample in φ-space and φ(s_t)."""
        o = obs
        op = next_obs
        a = as_tensor(actions, self.device)
        phi_t = self.icm._phi(o)
        phi_tp1 = self.icm._phi(op)
        a_fwd = self.icm._act_for_forward(a)
        pred = self.icm.forward_head(torch.cat([phi_t, a_fwd], dim=-1))
        per_dim = F.mse_loss(pred, phi_tp1, reduction="none")  # [B, D]
        return per_dim.mean(dim=-1), phi_t  # [B], [B, D]

    def compute(self, tr: Transition) -> IntrinsicOutput:
        """Single-sample LP (normalized) for the region of φ(s)."""
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
        """Vectorized LP for a batch; updates per-region EMAs.

        Samples are grouped by region id and EMAs are updated once per region
        using the mean error of samples mapped to that region. Per-sample LP
        within a region shares the same post-update value.
        """
        err_t, phi_t = self._forward_error_per_sample(obs, next_obs, actions)  # [B], [B,D]

        # Insert all points; get region ids
        phi_np = phi_t.detach().cpu().numpy()
        rids = self.store.bulk_insert(phi_np)  # (B,)
        err_np = err_t.detach().cpu().numpy().astype(np.float64)

        # Group errors by region id and compute per-region mean error
        uniq, inv = np.unique(rids, return_inverse=True)  # uniq[K], inv[B] in [0..K-1]
        sums = np.zeros(len(uniq), dtype=np.float64)
        cnts = np.zeros(len(uniq), dtype=np.int64)
        np.add.at(sums, inv, err_np)
        np.add.at(cnts, inv, 1)
        means = sums / np.maximum(1, cnts)

        # Update region stats once per region (EMA on the mean)
        lp_per_region = np.zeros(len(uniq), dtype=np.float32)
        for i, rid in enumerate(uniq):
            rid_i = int(rid)
            e_mean = float(means[i])
            st = self._stats.get(rid_i)
            if st is None or st.count == 0:
                # First observation(s) in this region: initialize EMAs to the mean error.
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

        # Broadcast per-region LP back to samples, normalize, and scale
        lp_arr = lp_per_region[inv].astype(np.float32)  # (B,)
        self._lp_rms.update(lp_arr)
        lp_norm = self._lp_rms.normalize(lp_arr)

        out = torch.as_tensor(self.alpha_lp * lp_norm, dtype=torch.float32, device=self.device)
        return out.mean() if reduction == "mean" else out

    # ----------------------------- losses & update ----------------------------

    def loss(self, obs: Any, next_obs: Any, actions: Any) -> dict[str, Tensor]:
        """ICM losses plus current normalized LP mean (diagnostic only)."""
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
        """Train ICM and report losses plus current normalized intrinsic mean."""
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

    # ----------------------- internal: per-region stats -----------------------

    def _update_region(self, rid: int, error: float) -> float:
        """Update EMAs for the region and return current LP ≥ 0."""
        st = self._stats.get(rid)
        if st is None or st.count == 0:
            self._stats[rid] = _RegionStats(ema_long=error, ema_short=error, count=1)
            return 0.0
        st.ema_long = self.beta_long * st.ema_long + (1.0 - self.beta_long) * error
        st.ema_short = self.beta_short * st.ema_short + (1.0 - self.beta_short) * error
        st.count += 1
        return max(0.0, st.ema_long - st.ema_short)

    # ----------------------------- diagnostics ----------------------------

    @property
    def lp_rms(self) -> float:
        """Current RMS used to normalize LP values."""
        return self._lp_rms.rms

    def export_diagnostics(self, out_dir: Path, step: int) -> None:
        """Append region stats to regions.jsonl and gates.csv (gates=1 for RIAC)."""
        _diag.export_diagnostics(self.store, self._stats, out_dir=out_dir, step=int(step))
