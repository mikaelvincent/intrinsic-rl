from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

_IGNORED_KEYS = {"profile_cuda_sync"}


def _json_stable(obj: Any) -> str:
    def _normalize(x: Any) -> Any:
        if isinstance(x, Mapping):
            return {k: _normalize(x[k]) for k in sorted(x.keys()) if k not in _IGNORED_KEYS}
        if isinstance(x, (list, tuple)):
            return [_normalize(v) for v in x]
        return x

    norm = _normalize(obj)
    return json.dumps(norm, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def compute_cfg_hash(cfg_like: Any) -> str:
    data = _json_stable(cfg_like)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]
