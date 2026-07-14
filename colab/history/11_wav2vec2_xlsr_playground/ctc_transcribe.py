"""Retrospective reconstruction; rerun one Wav2Vec2/XLS-R CTC checkpoint."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


DEFAULT_MODEL = "Jzuluaga/wav2vec2-large-960h-lv60-self-en-atc-uwb-atcc-and-atcosim"


def load_audio(path: Path, target_rate: int = 16_000) -> np.ndarray:
    audio, rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if rate != target_rate:
        ratio = Fraction(target_rate, rate)
        mono = resample_poly(mono, ratio.numerator, ratio.denominator).astype(np.float32)
    return mono


def run(path: Path, model_id: str) -> dict:
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    audio = load_audio(path)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inputs.input_values.to(device)).logits
    predicted = logits.argmax(dim=-1)
    text = processor.batch_decode(predicted)[0]
    return {
        "model_id": model_id,
        "audio": str(path),
        "samples_16k": len(audio),
        "logit_frames": logits.shape[1],
        "vocab_size": logits.shape[2],
        "text": text.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.audio, args.model)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
