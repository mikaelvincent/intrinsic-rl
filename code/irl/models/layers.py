from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn

__all__ = ["FlattenObs", "mlp"]


class FlattenObs(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x.view(1, -1)
        return x.view(x.size(0), -1)


def mlp(
    in_dim: int,
    hidden: Sequence[int] = (256, 256),
    out_dim: int | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = [FlattenObs()]
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, int(h)), nn.ReLU(inplace=True)]
        last = int(h)
    if out_dim is not None:
        layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)
