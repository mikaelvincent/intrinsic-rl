from __future__ import annotations


def infer_channels_hw(shape: tuple[int, ...]) -> tuple[int, tuple[int, int]]:
    if len(shape) != 3:
        if len(shape) >= 2:
            return int(shape[-1]), (int(shape[0]), int(shape[1]))
        return int(shape[-1]), (0, 0)

    c0 = int(shape[0])
    c2 = int(shape[-1])

    if c0 in (1, 3, 4) and c2 not in (1, 3, 4):
        return c0, (int(shape[1]), int(shape[2]))

    return c2, (int(shape[0]), int(shape[1]))
