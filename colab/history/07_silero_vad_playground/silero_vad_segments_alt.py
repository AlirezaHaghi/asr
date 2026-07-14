"""Retrospective reconstruction; results require rerunning.

Extract Silero timestamps from supplied audio. This is not an original
timestamped development artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad


TARGET_RATE = 16_000


def load_mono_16k(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate == TARGET_RATE:
        return mono
    old_x = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
    new_length = max(1, round(len(mono) * TARGET_RATE / rate))
    new_x = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
    return np.interp(new_x, old_x, mono).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--min-silence-ms", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("silero_segments.json"))
    args = parser.parse_args()
    try:
        audio = load_mono_16k(args.audio)
        model = load_silero_vad()
        timestamps = get_speech_timestamps(
            torch.from_numpy(audio.copy()),
            model,
            sampling_rate=TARGET_RATE,
            threshold=args.threshold,
            min_speech_duration_ms=args.min_speech_ms,
            min_silence_duration_ms=args.min_silence_ms,
            return_seconds=False,
        )
        segments = [
            {"start_sample": int(ts["start"]), "end_sample": int(ts["end"]), "start_s": round(ts["start"] / TARGET_RATE, 6), "end_s": round(ts["end"] / TARGET_RATE, 6)}
            for ts in timestamps
        ]
        payload = {"input": str(args.audio), "sample_rate": TARGET_RATE, "threshold": args.threshold, "segments": segments}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output} ({len(segments)} segments)")
        return 0
    except Exception:
        logging.exception("Silero VAD failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
