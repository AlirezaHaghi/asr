"""Retrospective reconstruction; rerun multiple Whisper checkpoints on one WAV."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path


DEFAULT_MODELS = [
    "jacktol/whisper-large-v3-finetuned-for-ATC",
    "tclin/whisper-large-v3-turbo-atcosim",
    "youngsangroh/whisper-small-atco2-atcosim",
]


def transcribe(model_id: str, audio: Path, beams: int, language: str) -> dict:
    import torch
    from transformers import pipeline

    cuda = torch.cuda.is_available()
    started = time.perf_counter()
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=0 if cuda else -1,
        torch_dtype=torch.float16 if cuda else torch.float32,
        chunk_length_s=30,
    )
    loaded_s = time.perf_counter() - started
    infer_started = time.perf_counter()
    output = pipe(
        str(audio),
        generate_kwargs={
            "language": language,
            "task": "transcribe",
            "temperature": 0.0,
            "num_beams": beams,
        },
    )
    infer_s = time.perf_counter() - infer_started
    result = {
        "model_id": model_id,
        "text": output["text"].strip(),
        "load_s": round(loaded_s, 3),
        "inference_s": round(infer_s, 3),
        "beams": beams,
    }
    del pipe
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--language", default="english")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for model_id in args.models:
        try:
            rows.append(transcribe(model_id, args.audio, args.beams, args.language))
        except Exception as exc:
            rows.append({"model_id": model_id, "error": f"{type(exc).__name__}: {exc}"})
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
