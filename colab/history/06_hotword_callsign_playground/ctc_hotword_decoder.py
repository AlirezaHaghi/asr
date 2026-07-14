"""Retrospective reconstruction; rerun to decode saved CTC logits with hotwords."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [token for token, _ in sorted(payload.items(), key=lambda pair: pair[1])]
    return list(payload)


def greedy(logits: np.ndarray, labels: list[str], blank_id: int = 0) -> str:
    token_ids = logits.argmax(axis=-1).tolist()
    collapsed = []
    previous = None
    for token_id in token_ids:
        if token_id != previous and token_id != blank_id:
            collapsed.append(labels[token_id])
        previous = token_id
    return "".join(collapsed).replace("|", " ").strip()


def beam_decode(logits: np.ndarray, labels: list[str], unigrams: list[str], beam: int) -> str:
    from pyctcdecode import build_ctcdecoder

    decoder = build_ctcdecoder(labels=labels, unigrams=unigrams or None)
    return decoder.decode(logits, beam_width=beam)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logits", type=Path)
    parser.add_argument("vocab", type=Path)
    parser.add_argument("--unigrams", type=Path)
    parser.add_argument("--beam", type=int, default=50)
    parser.add_argument("--blank-id", type=int, default=0)
    args = parser.parse_args()

    logits = np.load(args.logits)
    if logits.ndim == 3:
        logits = logits[0]
    labels = load_labels(args.vocab)
    words = args.unigrams.read_text(encoding="utf-8").splitlines() if args.unigrams else []
    print(json.dumps({
        "greedy": greedy(logits, labels, args.blank_id),
        "hotword_beam": beam_decode(logits, labels, words, args.beam),
        "beam": args.beam,
        "unigrams": len(words),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
