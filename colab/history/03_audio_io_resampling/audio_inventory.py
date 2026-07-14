"""Retrospective reconstruction from the final notebook/report; results require rerunning.

Recursively inventory WAV metadata and bounded signal statistics.
"""

# خودمونی: اسم WAV تضمین نمی‌کنه sample rate و channel درست باشه.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def inspect(path: Path, scan_seconds: float) -> dict:
    info = sf.info(path)
    frames = min(info.frames, max(1, int(scan_seconds * info.samplerate)))
    data, rate = sf.read(path, frames=frames, always_2d=True, dtype="float32")
    return {
        "path": str(path.resolve()), "samplerate": info.samplerate,
        "channels": info.channels, "frames": info.frames,
        "duration_s": info.duration, "format": info.format, "subtype": info.subtype,
        "scanned_frames": len(data), "finite": bool(np.isfinite(data).all()),
        "peak": float(np.max(np.abs(data))) if data.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--scan-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("audio_inventory.json"))
    args = parser.parse_args()
    rows, failures = [], []
    for path in sorted(args.root.rglob(args.pattern)):
        try:
            rows.append(inspect(path, args.scan_seconds))
        except Exception as exc:
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    rates = sorted({row["samplerate"] for row in rows})
    payload = {"root": str(args.root.resolve()), "files": rows, "failures": failures,
               "summary": {"count": len(rows), "sampling_rates": rates,
                           "total_duration_s": sum(row["duration_s"] for row in rows)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"inventoried {len(rows)} WAV files ({len(failures)} failures); wrote {args.output}")


if __name__ == "__main__":
    main()
