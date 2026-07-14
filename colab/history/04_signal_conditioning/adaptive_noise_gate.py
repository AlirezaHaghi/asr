"""Retrospective reconstruction from the final notebook/report; results require rerunning.

Apply a soft, frame-energy gate using an estimated noise floor.
"""

# خودمونی: صدای تمیزتر به گوش، لزوماً WER بهتر نیست.
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d


def gain_envelope(audio: np.ndarray, rate: int, frame_ms: float,
                  noise_percentile: float, margin_db: float, floor_gain: float) -> tuple[np.ndarray, dict]:
    mono = np.mean(audio, axis=1)
    frame = max(1, int(rate * frame_ms / 1000.0))
    pad = (-len(mono)) % frame
    padded = np.pad(mono, (0, pad))
    blocks = padded.reshape(-1, frame)
    rms = np.sqrt(np.mean(blocks**2, axis=1) + 1e-12)
    noise_rms = float(np.percentile(rms, noise_percentile))
    threshold = noise_rms * 10.0 ** (margin_db / 20.0)
    ratio = np.clip(rms / max(threshold, 1e-12), 0.0, 1.0)
    frame_gain = floor_gain + (1.0-floor_gain) * ratio**2
    sample_gain = np.repeat(frame_gain, frame)[:len(mono)]
    sample_gain = gaussian_filter1d(sample_gain, sigma=max(1.0, frame/4.0))
    return sample_gain, {"frame_samples": frame, "noise_rms": noise_rms,
                         "threshold_rms": threshold, "mean_gain": float(np.mean(sample_gain))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--noise-percentile", type=float, default=20.0)
    parser.add_argument("--margin-db", type=float, default=6.0)
    parser.add_argument("--floor-gain", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 <= args.noise_percentile <= 100 or not 0 <= args.floor_gain <= 1:
        parser.error("percentile must be [0,100] and floor gain [0,1]")
    audio, rate = sf.read(args.input, always_2d=True, dtype="float32")
    envelope, report = gain_envelope(audio, rate, args.frame_ms, args.noise_percentile,
                                     args.margin_db, args.floor_gain)
    gated = audio * envelope[:, None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, gated, rate, subtype="PCM_16")
    report.update(input=str(args.input), output=str(args.output), sampling_rate=rate)
    print(report)


if __name__ == "__main__":
    main()
