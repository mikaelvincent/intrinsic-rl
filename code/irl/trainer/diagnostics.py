from __future__ import annotations

from pathlib import Path
from typing import Any


def maybe_export_intrinsic_diagnostics(
    *,
    run_dir: Path,
    intrinsic_module: Any | None,
    method_l: str,
    use_intrinsic: bool,
    step: int,
    csv_interval: int,
) -> None:
    try:
        if intrinsic_module is None or not bool(use_intrinsic):
            return
        if str(method_l) != "riac":
            return
        if int(step) % int(csv_interval) != 0:
            return
        if not hasattr(intrinsic_module, "export_diagnostics"):
            return

        diag_dir = Path(run_dir) / "diagnostics"
        intrinsic_module.export_diagnostics(diag_dir, step=int(step))
    except Exception:
        return
