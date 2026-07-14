"""Retrospective reconstruction; results require rerunning on local audio.

Transcribe a single audio file with an ATC-finetuned Whisper model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import soundfile as sf
import torch
import transformers
from transformers import pipeline

DEFAULT_MODEL = "jacktol/whisper-large-v3-finetuned-for-ATC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--language", default="english")
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--chunk-length-s", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("transcript.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.beams < 1:
        raise ValueError("--beams must be >= 1")
    audio_info = sf.info(args.audio)
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    load_started = time.perf_counter()
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        torch_dtype=dtype,
        chunk_length_s=args.chunk_length_s,
    )
    load_s = time.perf_counter() - load_started
    generate_kwargs = {
        "language": args.language,
        "task": "transcribe",
        "temperature": args.temperature,
        "num_beams": args.beams,
    }
    infer_started = time.perf_counter()
    output = recognizer(
        str(args.audio),
        generate_kwargs=generate_kwargs,
        return_timestamps="word" if args.word_timestamps else True,
    )
    inference_s = time.perf_counter() - infer_started
    duration_s = float(audio_info.duration)
    payload = {
        "experiment": "retrospective_whisper_single_file",
        "audio": str(args.audio.resolve()),
        "audio_duration_s": round(duration_s, 6),
        "model_id": args.model,
        "generate_kwargs": generate_kwargs,
        "chunk_length_s": args.chunk_length_s,
        "text": output.get("text", "").strip(),
        "chunks": output.get("chunks", []),
        "load_s": round(load_s, 4),
        "inference_s": round(inference_s, 4),
        "real_time_factor": round(inference_s / duration_s, 4) if duration_s else None,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(payload["text"])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
