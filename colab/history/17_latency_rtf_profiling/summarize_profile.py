"""Retrospective reconstruction; results require rerunning.

Summarize one supplied profiling artifact. This is not an original timestamped
development artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import statistics


def percentile(values: list[float], probability: float) -> float | None:
    if not values: return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def stats(values: list[float]) -> dict:
    return {"count": len(values), "mean": statistics.fmean(values) if values else None, "median": statistics.median(values) if values else None, "p95": percentile(values, 0.95), "min": min(values) if values else None, "max": max(values) if values else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, default=Path("profile_summary.json"))
    args = parser.parse_args()
    try:
        payload = json.loads(args.profile.read_text(encoding="utf-8"))
        records = payload.get("records")
        if not isinstance(records, list): raise ValueError("profile has no records list")
        successful = [row for row in records if row.get("status") == "ok" and (row.get("duration_seconds") or 0) > 0]
        result = {"source": str(args.profile), "config": payload.get("config"), "config_sha256": payload.get("config_sha256"), "load_seconds": payload.get("load_seconds"), "files": {"total": len(records), "successful": len(successful), "failed": sum(row.get("status") == "error" for row in records)}, "wall_seconds": stats([float(row["wall_seconds"]) for row in successful]), "rtf": stats([float(row["rtf"]) for row in successful if row.get("rtf") is not None]), "audio_seconds": stats([float(row["duration_seconds"]) for row in successful])}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
        return 0
    except Exception:
        logging.exception("profile summary failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
