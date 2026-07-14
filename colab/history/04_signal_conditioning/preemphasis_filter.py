"""Retrospective reconstruction from the final notebook/report; results require rerunning.

Apply first-order pre-emphasis and optionally quantify inverse-filter error.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter


def preemphasize(audio: np.ndarray, coefficient: float) -> np.ndarray:
    return lfilter([1.0, -coefficient], [1.0], audio, axis=0)


def deemphasize(audio: np.ndarray, coefficient: float) -> np.ndarray:
    return lfilter([1.0], [1.0, -coefficient], audio, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--coefficient", type=float, default=0.97)
    parser.add_argument("--report-inverse-error", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.coefficient < 1.0:
        parser.error("--coefficient must be in [0, 1)")
    audio, rate = sf.read(args.input, always_2d=True, dtype="float32")
    filtered = preemphasize(audio, args.coefficient)
    peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0
    scale = min(1.0, 0.999 / peak) if peak else 1.0
    encoded = filtered * scale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, encoded, rate, subtype="PCM_16")
    report = {"coefficient": args.coefficient, "safety_scale": scale,
              "input_rms": float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0,
              "output_rms": float(np.sqrt(np.mean(encoded**2))) if encoded.size else 0.0}
    if args.report_inverse_error and audio.size:
        reconstructed = deemphasize(filtered, args.coefficient)
        report["inverse_rmse_float"] = float(np.sqrt(np.mean((reconstructed-audio)**2)))
    print(report)


if __name__ == "__main__":
    main()
