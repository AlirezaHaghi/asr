"""Retrospective reconstruction; results require rerunning.

Segment a PCM WAV file with a transparent frame-energy VAD. This is not an
original timestamped development artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import wave
from pathlib import Path

import numpy as np


def read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("Only 16-bit PCM WAV is supported by this script")
        rate = wav.getframerate()
        channels = wav.getnchannels()
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data / 32768.0, rate


def frame_dbfs(audio: np.ndarray, frame_samples: int) -> np.ndarray:
    if frame_samples <= 0:
        raise ValueError("frame_samples must be positive")
    count = max(1, int(np.ceil(len(audio) / frame_samples)))
    padded = np.pad(audio, (0, count * frame_samples - len(audio)))
    frames = padded.reshape(count, frame_samples)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def mask_to_segments(mask: np.ndarray, frame_samples: int, total: int, min_speech: int, hangover: int) -> list[dict]:
    segments: list[dict] = []
    start: int | None = None
    silent = 0
    for index, active in enumerate(mask.tolist() + [False] * (hangover + 1)):
        if active:
            start = index if start is None else start
            silent = 0
        elif start is not None:
            silent += 1
            if silent > hangover:
                end_frame = index - silent + 1
                if end_frame - start >= min_speech:
                    segments.append({"start_sample": start * frame_samples, "end_sample": min(total, end_frame * frame_samples)})
                start, silent = None, 0
    return segments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--threshold-db", type=float, default=-38.0)
    parser.add_argument("--frame-ms", type=float, default=30.0)
    parser.add_argument("--min-speech-ms", type=float, default=240.0)
    parser.add_argument("--hangover-ms", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=Path("energy_segments.json"))
    args = parser.parse_args()
    try:
        audio, rate = read_pcm16(args.wav)
        frame_samples = max(1, round(rate * args.frame_ms / 1000))
        levels = frame_dbfs(audio, frame_samples)
        min_frames = max(1, round(args.min_speech_ms / args.frame_ms))
        hangover = max(0, round(args.hangover_ms / args.frame_ms))
        segments = mask_to_segments(levels >= args.threshold_db, frame_samples, len(audio), min_frames, hangover)
        for item in segments:
            item["start_s"] = round(item["start_sample"] / rate, 6)
            item["end_s"] = round(item["end_sample"] / rate, 6)
        payload = {"input": str(args.wav), "sample_rate": rate, "threshold_db": args.threshold_db, "segments": segments}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output} ({len(segments)} segments)")
        return 0
    except Exception:
        logging.exception("energy VAD failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
