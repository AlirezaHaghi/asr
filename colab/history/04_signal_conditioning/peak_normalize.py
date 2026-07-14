"""Retrospective reconstruction from the final notebook/report; results require rerunning.

Apply transparent peak normalization with a chosen dBFS ceiling.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def normalize_peak(audio: np.ndarray, target_dbfs: float) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if not np.isfinite(peak) or peak == 0.0:
        return audio.copy(), 1.0
    target = 10.0 ** (target_dbfs / 20.0)
    gain = min(target / peak, 1.0 / peak)
    return np.clip(audio * gain, -1.0, 1.0), gain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-dbfs", type=float, default=-3.0)
    parser.add_argument("--mono", action="store_true")
    args = parser.parse_args()
    if args.target_dbfs > 0:
        parser.error("--target-dbfs must be <= 0")
    audio, rate = sf.read(args.input, always_2d=True, dtype="float32")
    if args.mono:
        audio = np.mean(audio, axis=1, keepdims=True)
    before = float(np.max(np.abs(audio))) if audio.size else 0.0
    output, gain = normalize_peak(audio, args.target_dbfs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, output, rate, subtype="PCM_16")
    print({"input_peak": before, "output_peak": float(np.max(np.abs(output))) if output.size else 0.0,
           "gain": gain, "gain_db": float(20*np.log10(gain)) if gain > 0 else None,
           "sampling_rate": rate, "output": str(args.output)})


if __name__ == "__main__":
    main()
