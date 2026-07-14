"""Retrospective reconstruction; results require a rerun.

Compare local ATC ASR observations on raw audio and Silero VAD segments.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from jiwer import cer, wer
from scipy.signal import resample_poly
from silero_vad import get_speech_timestamps, load_silero_vad
from transformers import pipeline

TARGET_SR = 16_000
DEFAULT_MODEL = "jacktol/whisper-large-v3-finetuned-for-ATC"


def load_mono(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sample_rate)
    return np.ascontiguousarray(audio, dtype=np.float32)


def transcribe(asr, segments: list[np.ndarray], beams: int) -> tuple[str, float]:
    started = time.perf_counter()
    texts = []
    for segment in segments:
        if segment.size < 400:
            continue
        output = asr(
            {"array": segment, "sampling_rate": TARGET_SR},
            generate_kwargs={"language": "english", "task": "transcribe", "temperature": 0.0, "num_beams": beams},
        )
        texts.append(output["text"].strip())
    return " ".join(texts).strip(), time.perf_counter() - started


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.5, 0.65])
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--reference", help="Optional reference transcript for observed WER/CER")
    parser.add_argument("--output", type=Path, default=Path("vad_asr_comparison.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    waveform = load_mono(args.audio)
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr = pipeline("automatic-speech-recognition", model=args.model, device=device, torch_dtype=dtype, chunk_length_s=30)
    vad_model = load_silero_vad()
    conditions: list[tuple[str, list[np.ndarray], dict]] = [("no_vad", [waveform], {"enabled": False})]
    for threshold in args.thresholds:
        timestamps = get_speech_timestamps(
            torch.from_numpy(waveform), vad_model, sampling_rate=TARGET_SR,
            threshold=threshold, min_speech_duration_ms=250, min_silence_duration_ms=100,
        )
        segments = [waveform[int(t["start"]):int(t["end"])] for t in timestamps] or [waveform]
        conditions.append((f"vad_{threshold:g}", segments, {"enabled": True, "threshold": threshold}))
    records = []
    for name, segments, config in conditions:
        transcript, elapsed = transcribe(asr, segments, args.beams)
        record = {
            "condition": name,
            "vad_config": config,
            "segments": len(segments),
            "transcript": transcript,
            "elapsed_s": round(elapsed, 4),
            "audio_s": round(waveform.size / TARGET_SR, 4),
            "real_time_factor": round(elapsed / (waveform.size / TARGET_SR), 4) if waveform.size else None,
        }
        if args.reference:
            ref, hyp = normalized(args.reference), normalized(transcript)
            record.update({"reference": args.reference, "wer": round(wer(ref, hyp), 6), "cer": round(cer(ref, hyp), 6)})
        records.append(record)
    payload = {"experiment": "retrospective_vad_asr_comparison", "model_id": args.model, "audio": str(args.audio.resolve()), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}; no result is valid until this command is rerun on local audio")


if __name__ == "__main__":
    main()
