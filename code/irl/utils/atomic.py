from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "atomic_replace",
    "atomic_write_text",
    "atomic_write_bytes",
    "atomic_write_csv",
]


def atomic_replace(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding=encoding, newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    atomic_replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    atomic_replace(tmp, path)


def atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
    *,
    extrasaction: str = "raise",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(fieldnames),
                extrasaction=str(extrasaction),
                restval="",
            )
            w.writeheader()
            for r in rows:
                if not isinstance(r, Mapping):
                    raise TypeError("atomic_write_csv: each row must be a mapping")
                w.writerow(r)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

        atomic_replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise
