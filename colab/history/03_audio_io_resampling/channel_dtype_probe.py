"""Retrospective reconstruction from the final notebook/report; results require rerunning.

Measure channel balance, amplitude, DC offset, and clipping risk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def channel_stats(values: np.ndarray) -> dict:
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0
    return {
        "peak": peak, "rms": rms, "dc_offset": float(np.mean(values)) if values.size else 0.0,
        "crest_factor": peak / rms if rms else None,
        "clipped_samples": int(np.count_nonzero(np.abs(values) >= 0.999)),
        "nonfinite_samples": int(np.count_nonzero(~np.isfinite(values))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("channel_probe.json"))
    args = parser.parse_args()
    info = sf.info(args.input)
    frames = -1 if args.max_seconds is None else int(max(0, args.max_seconds) * info.samplerate)
    data, rate = sf.read(args.input, frames=frames, always_2d=True, dtype="float32")
    correlation = np.corrcoef(data.T).tolist() if data.shape[1] > 1 and len(data) > 1 else [[1.0]]
    payload = {
        "path": str(args.input.resolve()), "sampling_rate": rate, "frames_analyzed": len(data),
        "file_subtype": info.subtype, "channels": [channel_stats(data[:, i]) for i in range(data.shape[1])],
        "channel_correlation": correlation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"analyzed {data.shape[1]} channel(s); wrote {args.output}")


if __name__ == "__main__":
    main()
