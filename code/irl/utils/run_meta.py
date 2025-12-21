from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from irl.utils.atomic import atomic_write_text


def _version_str(mod: object | None) -> str:
    if mod is None:
        return ""
    try:
        v = getattr(mod, "__version__", "")
    except Exception:
        return ""
    return str(v or "")


def collect_run_meta(*, device: str | torch.device | None = None, seed: int | None = None) -> dict[str, Any]:
    dev_s = ""
    if device is not None:
        try:
            dev_s = str(device).strip()
        except Exception:
            dev_s = ""

    gym_ver = ""
    try:
        import gymnasium as gym  # type: ignore

        gym_ver = _version_str(gym)
    except Exception:
        gym_ver = ""

    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": str(sys.version.split()[0]),
        "platform": str(platform.platform()),
        "pid": int(os.getpid()),
        "seed": None if seed is None else int(seed),
        "device": dev_s,
        "torch": _version_str(torch),
        "numpy": _version_str(np),
        "gymnasium": str(gym_ver),
        "torch_num_threads": int(torch.get_num_threads()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": str(getattr(torch.version, "cuda", "") or ""),
    }

    if bool(torch.cuda.is_available()):
        try:
            dev = torch.device(dev_s) if dev_s else torch.device("cuda")
            if dev.type == "cuda":
                idx = dev.index if dev.index is not None else int(torch.cuda.current_device())
                meta["cuda_device_index"] = int(idx)
                meta["cuda_name"] = str(torch.cuda.get_device_name(int(idx)))
        except Exception:
            pass

    return meta


def read_run_meta(path: Path) -> dict[str, Any] | None:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        data = json.loads(text)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_run_meta(path: Path, meta: Mapping[str, Any]) -> None:
    p = Path(path)
    payload = dict(meta)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write_text(p, text)
