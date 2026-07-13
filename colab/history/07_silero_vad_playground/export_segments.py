"""Retrospective reconstruction; results require a rerun.

Export Silero-detected speech regions and a reproducible JSONL manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from silero_vad import get_speech_timestamps, load_silero_vad

TARGET_SR = 16_000


def load_audio(path: Path) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = values.mean(axis=1)
    if sample_rate != TARGET_SR:
        mono = resample_poly(mono, TARGET_SR, sample_rate)
    return np.ascontiguousarray(mono, dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("segments"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--min-silence-ms", type=int, default=100)
    parser.add_argument("--speech-pad-ms", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    waveform = load_audio(args.audio)
    regions = get_speech_timestamps(
        torch.from_numpy(waveform),
        load_silero_vad(),
        sampling_rate=TARGET_SR,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_ms,
        min_silence_duration_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "segments.jsonl"
    records = []
    stem = args.audio.stem
    for index, region in enumerate(regions, start=1):
        start, end = int(region["start"]), int(region["end"])
        segment = waveform[start:end]
        destination = args.output_dir / f"{stem}_speech_{index:04d}.wav"
        sf.write(destination, segment, TARGET_SR, subtype="PCM_16")
        records.append(
            {
                "segment_id": f"{stem}_{index:04d}",
                "source_audio": str(args.audio.resolve()),
                "segment_path": str(destination.resolve()),
                "start_sample": start,
                "end_sample": end,
                "start_s": round(start / TARGET_SR, 6),
                "end_s": round(end / TARGET_SR, 6),
                "duration_s": round((end - start) / TARGET_SR, 6),
                "sha256": sha256_file(destination),
                "vad": {
                    "threshold": args.threshold,
                    "min_speech_ms": args.min_speech_ms,
                    "min_silence_ms": args.min_silence_ms,
                    "speech_pad_ms": args.speech_pad_ms,
                },
            }
        )
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"exported {len(records)} regions; manifest={manifest_path}")


if __name__ == "__main__":
    main()
