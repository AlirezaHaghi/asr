"""Retrospective reconstruction; rerun to build a unified ATC JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_spec(spec: str) -> tuple[str, str]:
    dataset_id, separator, split = spec.partition("::")
    return dataset_id, split if separator else "test"


def stable_id(dataset_id: str, split: str, index: int, text: str) -> str:
    raw = f"{dataset_id}|{split}|{index}|{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def rows_for(spec: str, limit: int):
    from datasets import load_dataset

    dataset_id, split = parse_spec(spec)
    ds = load_dataset(dataset_id, split=split, streaming=True)
    for index, item in enumerate(ds):
        if index >= limit:
            break
        text = next((item.get(k) for k in ("text", "transcription", "sentence") if item.get(k)), "")
        audio = item.get("audio", {})
        duration = None
        if isinstance(audio, dict) and audio.get("array") is not None and audio.get("sampling_rate"):
            duration = len(audio["array"]) / audio["sampling_rate"]
        yield {
            "id": item.get("id") or stable_id(dataset_id, split, index, str(text)),
            "dataset": dataset_id,
            "split": split,
            "text": str(text),
            "audio_path": audio.get("path") if isinstance(audio, dict) else None,
            "sampling_rate": audio.get("sampling_rate") if isinstance(audio, dict) else None,
            "duration_s": round(duration, 3) if duration is not None else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, help="id::split")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for spec in args.dataset:
            for row in rows_for(spec, args.limit):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
