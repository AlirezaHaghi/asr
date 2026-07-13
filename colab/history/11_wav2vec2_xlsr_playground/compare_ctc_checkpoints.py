"""Retrospective reconstruction; rerun several CTC checkpoints on the same WAV."""

from __future__ import annotations

import argparse
import gc
import json
import time
from fractions import Fraction
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly


def audio_16k(path: Path):
    data, rate = sf.read(path, always_2d=True, dtype="float32")
    data = data.mean(axis=1)
    if rate != 16_000:
        ratio = Fraction(16_000, rate)
        data = resample_poly(data, ratio.numerator, ratio.denominator)
    return data


def one(model_id: str, audio) -> dict:
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    load_s = time.perf_counter() - started
    inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
    infer_started = time.perf_counter()
    with torch.inference_mode():
        logits = model(inputs.input_values.to(device)).logits
    text = processor.batch_decode(logits.argmax(dim=-1))[0]
    result = {"model_id": model_id, "text": text.strip(), "load_s": round(load_s, 3), "inference_s": round(time.perf_counter() - infer_started, 3)}
    del model, processor, logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).with_name("ctc_model_catalog.json"))
    parser.add_argument("--model", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    models = args.model or [row["model_id"] for row in json.loads(args.catalog.read_text(encoding="utf-8"))]
    audio = audio_16k(args.audio)
    rows = []
    for model_id in models:
        try:
            rows.append(one(model_id, audio))
        except Exception as exc:  # یه مدل خراب شد، بقیه رو ول نمی‌کنیم
            rows.append({"model_id": model_id, "error": f"{type(exc).__name__}: {exc}"})
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
