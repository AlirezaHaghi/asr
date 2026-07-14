"""Retrospective reconstruction; rerun to build an ATC callsign lexicon."""

# خودمونی: hotword یه هل کوچیکه، نه جواب آماده برای مدل.

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_AIRLINES = [
    "speedbird", "ryanair", "eurowings", "lufthansa", "airfrance",
    "easyjet", "belavia", "turkish", "austrian", "swiss", "klm",
]


def clean_phrase(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 -]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def variants(word: str) -> set[str]:
    compact = word.replace(" ", "").replace("-", "")
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", word).lower()
    return {clean_phrase(word), clean_phrase(compact), clean_phrase(spaced)} - {""}


def read_words(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_AIRLINES
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--transcripts", type=Path, help="optional text/JSONL corpus")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = read_words(args.input)
    vocabulary = sorted({item for word in source for item in variants(word)})
    counts: Counter[str] = Counter()
    if args.transcripts:
        corpus = args.transcripts.read_text(encoding="utf-8").lower()
        for term in vocabulary:
            counts[term] = len(re.findall(rf"\b{re.escape(term)}\b", corpus))

    report = {
        "source_terms": len(source),
        "variants": vocabulary,
        "corpus_counts": dict(counts),
        "unigrams": sorted(vocabulary, key=lambda item: (-counts[item], item)),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
