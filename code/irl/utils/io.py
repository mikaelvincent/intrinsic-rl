from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .atomic_files import atomic_replace


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
