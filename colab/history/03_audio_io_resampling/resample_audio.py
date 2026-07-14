"""Retrospective reconstruction from the final notebook/report; results require rerunning.

Resample an audio file with a rational polyphase filter.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return data.copy()
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(data, target_rate // divisor, source_rate // divisor, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-rate", type=int, default=16000)
    parser.add_argument("--mono", action="store_true")
    parser.add_argument("--subtype", default="PCM_16", choices=("PCM_16", "PCM_24", "FLOAT"))
    args = parser.parse_args()
    if args.target_rate <= 0:
        parser.error("--target-rate must be positive")
    data, source_rate = sf.read(args.input, always_2d=True, dtype="float32")
    if args.mono:
        data = np.mean(data, axis=1, keepdims=True)
    converted = np.clip(resample(data, source_rate, args.target_rate), -1.0, 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, converted, args.target_rate, subtype=args.subtype)
    print({"input_rate": source_rate, "output_rate": args.target_rate,
           "input_frames": len(data), "output_frames": len(converted),
           "channels": converted.shape[1], "output": str(args.output)})


if __name__ == "__main__":
    main()
