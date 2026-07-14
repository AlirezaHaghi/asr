"""Retrospective reconstruction; rerun to compare ATC dataset schemas and samples."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter


def parse_spec(spec: str) -> tuple[str, str]:
    dataset_id, separator, split = spec.partition("::")
    return dataset_id, split if separator else "test"


def summarize(spec: str, limit: int) -> dict:
    from datasets import load_dataset

    dataset_id, split = parse_spec(spec)
    ds = load_dataset(dataset_id, split=split, streaming=True)
    columns: Counter[str] = Counter()
    text_lengths: list[int] = []
    durations: list[float] = []
    examples: list[dict] = []

    for index, item in enumerate(ds):
        if index >= limit:
            break
        columns.update(item.keys())
        text = next((item.get(k) for k in ("text", "transcription", "sentence") if item.get(k)), "")
        text_lengths.append(len(str(text).split()))
        audio = item.get("audio")
        if isinstance(audio, dict) and audio.get("array") is not None and audio.get("sampling_rate"):
            durations.append(len(audio["array"]) / audio["sampling_rate"])
        if len(examples) < 3:
            examples.append({"text": str(text)[:180], "keys": sorted(item.keys())})

    return {
        "dataset": dataset_id,
        "split": split,
        "sampled": len(text_lengths),
        "columns_seen": dict(columns),
        "median_words": statistics.median(text_lengths) if text_lengths else None,
        "median_duration_s": statistics.median(durations) if durations else None,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, help="id::split")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = [summarize(spec, args.limit) for spec in args.dataset]
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
