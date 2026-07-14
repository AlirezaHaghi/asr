"""Retrospective reconstruction; rerun base and ATC-fine-tuned Parakeet fairly."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path


MODELS = [
    "nvidia/parakeet-tdt-0.6b-v3",
    "qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC",
]


def one(model_id: str, audio: Path) -> dict:
    import torch
    from nemo.collections.asr.models import ASRModel

    started = time.perf_counter()
    model = ASRModel.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.cuda()
    load_s = time.perf_counter() - started
    infer_started = time.perf_counter()
    hyp = model.transcribe([str(audio)])[0]
    result = {
        "model_id": model_id,
        "text": str(getattr(hyp, "text", hyp)),
        "load_s": round(load_s, 3),
        "inference_s": round(time.perf_counter() - infer_started, 3),
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for model_id in args.model or MODELS:
        try:
            rows.append(one(model_id, args.audio))
        except Exception as exc:
            rows.append({"model_id": model_id, "error": f"{type(exc).__name__}: {exc}"})
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
