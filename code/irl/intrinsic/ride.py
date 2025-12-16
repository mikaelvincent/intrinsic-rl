from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import gymnasium as gym
import torch
from torch import Tensor, nn

from irl.utils.torchops import as_tensor

from . import BaseIntrinsicModule, IntrinsicOutput, Transition
from .icm import ICM, ICMConfig


class RIDE(BaseIntrinsicModule, nn.Module):
    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        device: Union[str, torch.device] = "cpu",
        icm: Optional[ICM] = None,
        icm_cfg: Optional[ICMConfig] = None,
        *,
        bin_size: float = 0.25,
        alpha_impact: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("RIDE supports Box observation spaces (vector or image).")
        self.device = torch.device(device)

        self.icm = icm if icm is not None else ICM(obs_space, act_space, device=device, cfg=icm_cfg)
        self.encoder = self.icm.encoder
        self.is_discrete = self.icm.is_discrete

        self.bin_size = float(bin_size)
        self.alpha_impact = float(alpha_impact)

        self._nvec: Optional[int] = None
        self._ep_counts: list[dict[Tuple[int, ...], int]] = []

        self.to(self.device)

    @torch.no_grad()
    def _impact_per_sample(self, obs: Any, next_obs: Any) -> Tensor:
        phi_t = self.icm._phi(obs)
        phi_tp1 = self.icm._phi(next_obs)
        return torch.norm(phi_tp1 - phi_t, p=2, dim=-1)

    def _ensure_counts(self, batch_size: int) -> None:
        if self._nvec is None or self._nvec != int(batch_size):
            self._nvec = int(batch_size)
            self._ep_counts = [dict() for _ in range(self._nvec)]

    @torch.no_grad()
    def _bin_keys(self, phi: Tensor) -> list[Tuple[int, ...]]:
        bins = torch.floor(phi / float(self.bin_size)).to(dtype=torch.int64, device="cpu")
        return [tuple(map(int, row.tolist())) for row in bins]

    def compute(self, tr: Transition) -> IntrinsicOutput:
        with torch.no_grad():
            r_raw = self._impact_per_sample(tr.s, tr.s_next).view(-1)[0]
            r = self.alpha_impact * r_raw
            return IntrinsicOutput(r_int=float(r.item()))

    def compute_batch(
        self, obs: Any, next_obs: Any, actions: Any | None = None, reduction: str = "none"
    ) -> Tensor:
        with torch.no_grad():
            r = self._impact_per_sample(obs, next_obs)
            r = self.alpha_impact * r
            return r.mean() if reduction == "mean" else r

    @torch.no_grad()
    def compute_impact_binned(
        self, obs: Any, next_obs: Any, dones: Any | None = None, reduction: str = "none"
    ) -> Tensor:
        phi_t = self.icm._phi(obs)
        phi_tp1 = self.icm._phi(next_obs)
        B = int(phi_tp1.size(0))
        self._ensure_counts(B)

        if dones is not None:
            d = as_tensor(dones, device=torch.device("cpu"), dtype=torch.float32).view(-1)
            for i in range(B):
                if bool(d[i].item()):
                    self._ep_counts[i].clear()

        keys = self._bin_keys(phi_tp1)
        raw = torch.norm(phi_tp1 - phi_t, p=2, dim=-1).to(device=self.device)

        out = torch.empty_like(raw)
        for i in range(B):
            cnt = self._ep_counts[i].get(keys[i], 0)
            denom = 1.0 + float(cnt)
            out[i] = (self.alpha_impact * raw[i]) / denom
            self._ep_counts[i][keys[i]] = cnt + 1

        return out.mean() if reduction == "mean" else out

    def loss(self, obs: Any, next_obs: Any, actions: Any) -> dict[str, Tensor]:
        icm_losses = self.icm.loss(obs, next_obs, actions)
        with torch.no_grad():
            r = self._impact_per_sample(obs, next_obs).mean()
        return {
            "total": icm_losses["total"],
            "icm_forward": icm_losses["forward"],
            "icm_inverse": icm_losses["inverse"],
            "intrinsic_mean": r,
        }

    def update(self, obs: Any, next_obs: Any, actions: Any, steps: int = 1) -> dict[str, float]:
        with torch.no_grad():
            r_mean = self._impact_per_sample(obs, next_obs).mean().detach().item()

        metrics = self.icm.update(obs, next_obs, actions, steps=steps)
        return {
            "loss_total": float(metrics["loss_total"]),
            "loss_forward": float(metrics["loss_forward"]),
            "loss_inverse": float(metrics["loss_inverse"]),
            "intrinsic_mean": float(r_mean),
        }
