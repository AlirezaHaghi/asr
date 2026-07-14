"""Retrospective reconstruction from the final notebook/report; results require a rerun.

Build a bounded JSONL manifest from decoded ATCO2 rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "Jzuluaga/atco2_corpus_1h"


def manifest_row(item: dict, index: int) -> dict:
    text_key = "text" if "text" in item else "transcription"
    audio = item.get("audio") or {}
    samples = audio.get("array")
    rate = int(audio.get("sampling_rate") or 0)
    count = int(len(samples)) if samples is not None else 0
    return {
        "index": index,
        "id": item.get("id", f"sample_{index:04d}"),
        "text_key": text_key,
        "transcript": str(item.get(text_key, "")).strip(),
        "sampling_rate": rate,
        "num_samples": count,
        "duration_s": round(count / rate, 4) if rate else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("manifest.jsonl"))
    args = parser.parse_args()
    ds = load_dataset(DATASET_ID, split=args.split, streaming=args.streaming)
    rows = []
    for index, item in enumerate(ds):
        if index >= args.limit:
            break
        rows.append(manifest_row(item, index))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    total_s = sum(row["duration_s"] or 0.0 for row in rows)
    print(f"wrote {len(rows)} rows ({total_s:.1f}s decoded audio) to {args.output}")


if __name__ == "__main__":
    main()
