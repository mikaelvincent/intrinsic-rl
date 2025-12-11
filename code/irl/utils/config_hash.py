"""Deterministic configuration hashing utilities for resume safety."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def _json_stable(obj: Any) -> str:
    """Return a stable JSON string suitable for hashing.

    The representation uses sorted keys at all nested levels and compact
    separators so that the resulting string is deterministic for a given
    logical mapping.
    """

    # Recursively sort dict keys and convert non-JSON types where practical.
    def _normalize(x: Any) -> Any:
        if isinstance(x, Mapping):
            return {k: _normalize(x[k]) for k in sorted(x.keys())}
        if isinstance(x, (list, tuple)):
            return [_normalize(v) for v in x]
        # primitives (including numpy scalars will be stringified by json if unsupported)
        return x

    norm = _normalize(obj)
    return json.dumps(norm, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def compute_cfg_hash(cfg_like: Any) -> str:
    """Compute a short SHA-256 hash (first 16 hex chars) of a config mapping.

    The hash is based on a stable JSON representation with sorted keys to make
    it invariant to Python dict insertion order differences.

    Parameters
    ----------
    cfg_like :
        A mapping (for example, the dict produced by ``irl.cfg.to_dict``)
        or any JSON-serialisable object.

    Returns
    -------
    str
        Lowercase hex digest prefix (16 characters) of the SHA-256 hash.
    """
    data = _json_stable(cfg_like)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]
