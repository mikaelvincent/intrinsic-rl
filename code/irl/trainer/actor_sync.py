from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from irl.models import PolicyNetwork


@dataclass
class ActorPolicySync:
    actor_policy: Optional[PolicyNetwork] = None

    @classmethod
    def maybe_create(
        cls,
        *,
        obs_space: object,
        act_space: object,
        device: torch.device,
        logger: object,
    ) -> "ActorPolicySync":
        if device.type == "cpu":
            return cls(actor_policy=None)

        actor_policy = PolicyNetwork(obs_space, act_space).to(torch.device("cpu"))
        actor_policy.eval()
        for p in actor_policy.parameters():
            p.requires_grad = False

        if hasattr(logger, "info"):
            logger.info("Using CPU actor policy for env stepping; syncing each PPO update.")

        return cls(actor_policy=actor_policy)

    def sync_from(self, learner_policy: PolicyNetwork) -> None:
        if self.actor_policy is None:
            return
        with torch.no_grad():
            sd = learner_policy.state_dict()
            cpu_sd = {k: v.detach().to("cpu") for k, v in sd.items()}
            self.actor_policy.load_state_dict(cpu_sd, strict=True)
