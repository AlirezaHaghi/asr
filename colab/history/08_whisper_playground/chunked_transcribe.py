"""Retrospective reconstruction; results require rerunning on local audio.

Study explicit overlapping-window Whisper inference without hiding merge ambiguity.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import pipeline

TARGET_SR = 16_000
DEFAULT_MODEL = "jacktol/whisper-large-v3-finetuned-for-ATC"


def load_mono_16k(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != TARGET_SR:
        mono = resample_poly(mono, TARGET_SR, sample_rate)
    return np.ascontiguousarray(mono, dtype=np.float32)


def windows(total: int, chunk: int, overlap: int):
    if chunk <= 0 or overlap < 0 or overlap >= chunk:
        raise ValueError("require chunk > 0 and 0 <= overlap < chunk")
    step = chunk - overlap
    start = 0
    while start < total:
        end = min(start + chunk, total)
        yield start, end
        if end == total:
            break
        start += step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("chunked.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    waveform = load_mono_16k(args.audio)
    chunk_samples = round(args.chunk_seconds * TARGET_SR)
    overlap_samples = round(args.overlap_seconds * TARGET_SR)
    device = 0 if torch.cuda.is_available() else -1
    recognizer = pipeline(
        "automatic-speech-recognition", model=args.model, device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    records = []
    for index, (start, end) in enumerate(windows(waveform.size, chunk_samples, overlap_samples), start=1):
        started = time.perf_counter()
        output = recognizer(
            {"array": waveform[start:end], "sampling_rate": TARGET_SR},
            generate_kwargs={"language": "english", "task": "transcribe", "temperature": 0.0, "num_beams": args.beams},
            return_timestamps=True,
        )
        records.append(
            {
                "chunk_index": index,
                "start_s": round(start / TARGET_SR, 6),
                "end_s": round(end / TARGET_SR, 6),
                "text": output.get("text", "").strip(),
                "timestamps": output.get("chunks", []),
                "inference_s": round(time.perf_counter() - started, 4),
            }
        )
    payload = {
        "experiment": "retrospective_explicit_chunking",
        "audio": str(args.audio.resolve()),
        "model_id": args.model,
        "chunk_seconds": args.chunk_seconds,
        "overlap_seconds": args.overlap_seconds,
        "merge_policy": "no automatic de-duplication; inspect adjacent chunks",
        "naive_joined_text": " ".join(r["text"] for r in records),
        "chunks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(records)} observed chunks to {args.output}")


if __name__ == "__main__":
    main()
