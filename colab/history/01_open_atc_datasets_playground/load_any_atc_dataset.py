"""Retrospective reconstruction; rerun to inspect an HF or local ATC dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


TEXT_CANDIDATES = ("text", "transcription", "sentence", "transcript")
AUDIO_CANDIDATES = ("audio", "path", "file", "audio_path")


def pick_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {name.lower(): name for name in columns}
    return next((lowered[name] for name in candidates if name in lowered), None)


def compact(value: Any) -> Any:
    if isinstance(value, dict) and "array" in value:
        array = value.get("array")
        return {
            "sampling_rate": value.get("sampling_rate"),
            "samples": len(array) if array is not None else None,
            "path": value.get("path"),
        }
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": list(value.shape)}
    if isinstance(value, str) and len(value) > 300:
        return value[:297] + "..."
    return value


def load_local(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else payload.get("records", [payload])


def load_hf(dataset_id: str, split: str, config: str | None, streaming: bool):
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": streaming}
    if config:
        return load_dataset(dataset_id, config, **kwargs)
    return load_dataset(dataset_id, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="HF dataset id or local JSON/JSONL")
    parser.add_argument("--split", default="test")
    parser.add_argument("--config")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    local = Path(args.source)
    dataset = load_local(local) if local.exists() else load_hf(
        args.source, args.split, args.config, args.streaming
    )
    sample = []
    for index, row in enumerate(dataset):
        if index >= args.limit:
            break
        sample.append({key: compact(value) for key, value in dict(row).items()})

    columns = list(sample[0]) if sample else []
    report = {
        "source": args.source,
        "split": args.split,
        "columns": columns,
        "text_column_guess": pick_column(columns, TEXT_CANDIDATES),
        "audio_column_guess": pick_column(columns, AUDIO_CANDIDATES),
        "sample": sample,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
