"""Retrospective reconstruction; results require a rerun.

Sweep Silero VAD settings on one local audio file and save observed regions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from silero_vad import get_speech_timestamps, load_silero_vad

TARGET_SR = 16_000


@dataclass(frozen=True)
class AudioInfo:
    path: str
    source_sample_rate: int
    samples_16k: int
    duration_s: float
    peak: float
    rms: float


def load_mono_16k(path: Path) -> tuple[np.ndarray, AudioInfo]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != TARGET_SR:
        mono = resample_poly(mono, TARGET_SR, sample_rate).astype(np.float32)
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    info = AudioInfo(
        path=str(path.resolve()),
        source_sample_rate=int(sample_rate),
        samples_16k=int(mono.size),
        duration_s=round(mono.size / TARGET_SR, 6),
        peak=round(float(np.max(np.abs(mono), initial=0.0)), 8),
        rms=round(float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0, 8),
    )
    return mono, info


def summarize_regions(regions: list[dict[str, int]], sample_count: int) -> dict:
    durations = [(r["end"] - r["start"]) / TARGET_SR for r in regions]
    speech_s = float(sum(durations))
    return {
        "segment_count": len(regions),
        "speech_s": round(speech_s, 6),
        "speech_fraction": round(speech_s / (sample_count / TARGET_SR), 6) if sample_count else 0.0,
        "mean_segment_s": round(float(np.mean(durations)), 6) if durations else 0.0,
        "median_segment_s": round(float(np.median(durations)), 6) if durations else 0.0,
        "regions": [
            {
                "start_sample": int(r["start"]),
                "end_sample": int(r["end"]),
                "start_s": round(r["start"] / TARGET_SR, 6),
                "end_s": round(r["end"] / TARGET_SR, 6),
            }
            for r in regions
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--min-silence-ms", type=int, default=100)
    parser.add_argument("--speech-pad-ms", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("vad_sweep.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for value in args.thresholds:
        if not 0.0 < value < 1.0:
            raise ValueError(f"threshold must be in (0, 1): {value}")
    waveform, audio_info = load_mono_16k(args.audio)
    model = load_silero_vad()
    tensor = torch.from_numpy(waveform)
    runs = []
    for threshold in args.thresholds:
        regions = get_speech_timestamps(
            tensor,
            model,
            sampling_rate=TARGET_SR,
            threshold=threshold,
            min_speech_duration_ms=args.min_speech_ms,
            min_silence_duration_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
        )
        runs.append(
            {
                "threshold": threshold,
                "min_speech_duration_ms": args.min_speech_ms,
                "min_silence_duration_ms": args.min_silence_ms,
                "speech_pad_ms": args.speech_pad_ms,
                **summarize_regions(regions, waveform.size),
            }
        )
    payload = {"experiment": "retrospective_vad_sweep", "audio": asdict(audio_info), "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output} with {len(runs)} observed configurations")


if __name__ == "__main__":
    main()
