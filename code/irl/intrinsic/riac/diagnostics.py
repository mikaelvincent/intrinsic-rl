"""Diagnostics helpers for RIAC: write regions.jsonl and gates.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import csv
import json

from irl.intrinsic.regions import KDTreeRegionStore


def _region_records(store: KDTreeRegionStore, stats: Dict[int, Any], step: int) -> List[dict]:
    """Build a list of JSON-serializable records for current regions.

    Parameters
    ----------
    store : KDTreeRegionStore
        Region store used by the RIAC module.
    stats : dict[int, Any]
        Mapping from region id to per-region statistics objects. The
        objects are expected to expose ``ema_long`` and ``ema_short``
        attributes and a ``count`` field.
    step : int
        Training step associated with this snapshot.

    Returns
    -------
    list[dict]
        One record per leaf region describing counts, EMAs, learning
        progress, gate state, and bounding boxes.
    """
    recs: List[dict] = []
    for leaf in store.iter_leaves():
        rid = int(leaf.region_id) if leaf.region_id is not None else -1
        st = stats.get(rid)
        ema_l = None if st is None else float(st.ema_long)
        ema_s = None if st is None else float(st.ema_short)
        lp = 0.0
        if st is not None and st.count > 0:
            lp = max(0.0, float(st.ema_long - st.ema_short))
        recs.append(
            {
                "step": int(step),
                "region_id": rid,
                "depth": int(leaf.depth),
                "count_leaf": int(leaf.count),
                "ema_long": ema_l,
                "ema_short": ema_s,
                "lp": float(lp),
                "gate": 1,  # RIAC has no gating; keep 1 for compatibility
                "bbox_lo": (
                    None if leaf.bbox_lo is None else [float(x) for x in leaf.bbox_lo.tolist()]
                ),
                "bbox_hi": (
                    None if leaf.bbox_hi is None else [float(x) for x in leaf.bbox_hi.tolist()]
                ),
            }
        )
    return recs


def export_diagnostics(
    store: KDTreeRegionStore, stats: Dict[int, Any], out_dir: Path, step: int
) -> None:
    """Append region statistics to ``regions.jsonl`` and ``gates.csv``.

    Parameters
    ----------
    store : KDTreeRegionStore
        Region store that defines the current set of leaf regions.
    stats : dict[int, Any]
        Mapping from region id to statistics objects used to compute
        learning progress.
    out_dir : pathlib.Path
        Directory where diagnostic files will be created if necessary
        and subsequently appended to.
    step : int
        Global training step to record alongside each diagnostic row.

    Notes
    -----
    The function appends one JSON record per region to
    ``regions.jsonl`` and a corresponding row to ``gates.csv`` with
    columns ``step``, ``region_id``, and ``gate``. For RIAC, the gate
    value is always ``1`` to remain compatible with the Proposed
    intrinsic diagnostics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    regions_path = out_dir / "regions.jsonl"
    gates_path = out_dir / "gates.csv"

    recs = _region_records(store, stats, step=int(step))

    # JSONL append
    with regions_path.open("a", encoding="utf-8") as f_jsonl:
        for r in recs:
            json.dump(r, f_jsonl, ensure_ascii=False)
            f_jsonl.write("\n")

    # CSV append (header once)
    write_header = not gates_path.exists() or gates_path.stat().st_size == 0
    with gates_path.open("a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(["step", "region_id", "gate"])
        for r in recs:
            writer.writerow([int(r["step"]), int(r["region_id"]), int(r["gate"])])
