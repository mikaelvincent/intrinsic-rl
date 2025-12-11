"""Atomic file write helpers (text/bytes) used across the project."""

from __future__ import annotations

import os
from pathlib import Path


def _atomic_replace(src: Path, dst: Path) -> None:
    """Atomically replace ``dst`` by ``src`` where supported."""
    os.replace(src, dst)


def atomic_replace(src: Path, dst: Path) -> None:
    """Atomically replace ``dst`` with ``src`` on the local filesystem."""
    _atomic_replace(Path(src), Path(dst))


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Atomically write a text file using a temporary file and fsync.

    Parameters
    ----------
    path :
        Destination file path.
    text :
        Text content to write.
    encoding :
        Text encoding, default is UTF-8.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit fsync for durability
    with open(tmp, "w", encoding=encoding, newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    _atomic_replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write a binary file using a temporary file and fsync."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    _atomic_replace(tmp, path)
