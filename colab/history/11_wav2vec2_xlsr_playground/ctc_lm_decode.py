"""Retrospective reconstruction; rerun CTC greedy and optional KenLM decoding."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kenlm", type=Path)
    parser.add_argument("--unigrams", type=Path)
    parser.add_argument("--beam", type=int, default=100)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    data, rate = sf.read(args.audio, always_2d=True, dtype="float32")
    data = data.mean(axis=1)
    if rate != 16_000:
        ratio = Fraction(16_000, rate)
        data = resample_poly(data, ratio.numerator, ratio.denominator)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model).eval()
    inputs = processor(data, sampling_rate=16_000, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inputs.input_values).logits[0].numpy()
    greedy = processor.decode(logits.argmax(axis=-1))
    report = {"model_id": args.model, "greedy": greedy, "beam": None}

    if args.kenlm:
        from pyctcdecode import build_ctcdecoder

        vocab = processor.tokenizer.get_vocab()
        labels = [token for token, _ in sorted(vocab.items(), key=lambda pair: pair[1])]
        unigrams = args.unigrams.read_text(encoding="utf-8").splitlines() if args.unigrams else None
        decoder = build_ctcdecoder(labels, kenlm_model_path=str(args.kenlm), unigrams=unigrams)
        report["beam"] = decoder.decode(logits, beam_width=args.beam)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
