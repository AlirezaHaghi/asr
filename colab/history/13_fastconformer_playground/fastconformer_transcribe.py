"""Retrospective reconstruction; rerun FastConformer on supplied WAV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_text(item) -> str:
    if isinstance(item, str):
        return item
    return str(getattr(item, "text", item))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--model", default="niclaswue/youtube-atc-fastconformer")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from nemo.collections.asr.models import ASRModel

    model = ASRModel.from_pretrained(args.model)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    paths = [str(path) for path in args.audio]
    hypotheses = model.transcribe(paths, batch_size=args.batch_size)
    rows = [
        {"file": path, "text": extract_text(hypothesis), "model_id": args.model}
        for path, hypothesis in zip(paths, hypotheses)
    ]
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
