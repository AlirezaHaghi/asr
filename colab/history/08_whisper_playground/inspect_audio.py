"""Retrospective reconstruction; results require rerunning on local audio.

Inspect one audio file without loading an ASR model and write an auditable profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dbfs(value: float) -> float | None:
    return round(20.0 * np.log10(value), 3) if value > 0 else None


def profile_audio(path: Path, silence_threshold: float) -> dict:
    info = sf.info(path)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    peak = float(np.max(np.abs(mono), initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    clipped = np.count_nonzero(np.abs(audio) >= 0.999)
    near_silent = np.count_nonzero(np.abs(mono) < silence_threshold)
    channel_rms = np.sqrt(np.mean(np.square(audio), axis=0)) if audio.size else np.zeros(audio.shape[1])
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "container_format": info.format,
        "subtype": info.subtype,
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "frames": int(audio.shape[0]),
        "duration_s": round(audio.shape[0] / sample_rate, 6),
        "peak": round(peak, 8),
        "peak_dbfs": dbfs(peak),
        "rms": round(rms, 8),
        "rms_dbfs": dbfs(rms),
        "channel_rms": [round(float(v), 8) for v in channel_rms],
        "clipped_sample_values": int(clipped),
        "near_silent_fraction": round(near_silent / mono.size, 6) if mono.size else 0.0,
        "silence_proxy_threshold": silence_threshold,
        "whisper_compatibility": {
            "already_mono": audio.shape[1] == 1,
            "already_16khz": sample_rate == 16_000,
            "conversion_needed": audio.shape[1] != 1 or sample_rate != 16_000,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--silence-threshold", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("audio_profile.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {"experiment": "retrospective_single_file_profile", **profile_audio(args.audio, args.silence_threshold)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote observed profile to {args.output}")


if __name__ == "__main__":
    main()
