"""Retrospective reconstruction from the final notebook/report; results require a rerun.

Inspect the ATCO2 Hugging Face dataset schema and a bounded sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_ID = "Jzuluaga/atco2_corpus_1h"


def summarize(value: Any) -> Any:
    if isinstance(value, dict):
        result = {"keys": sorted(value.keys())}
        if "sampling_rate" in value:
            result["sampling_rate"] = value["sampling_rate"]
        if "array" in value:
            array = value["array"]
            result["array_shape"] = list(getattr(array, "shape", [len(array)]))
            result["array_dtype"] = str(getattr(array, "dtype", type(array).__name__))
        return result
    if isinstance(value, str):
        return {"type": "str", "length": len(value), "preview": value[:160]}
    return {"type": type(value).__name__, "preview": repr(value)[:160]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("schema.json"))
    args = parser.parse_args()
    ds = load_dataset(DATASET_ID, split=args.split, streaming=args.streaming)
    examples = []
    for index, item in enumerate(ds):
        if index >= args.limit:
            break
        examples.append({key: summarize(value) for key, value in item.items()})
    payload = {
        "dataset_id": DATASET_ID,
        "split": args.split,
        "streaming": args.streaming,
        "features": str(getattr(ds, "features", None)),
        "column_names": list(getattr(ds, "column_names", []) or (examples[0].keys() if examples else [])),
        "observed_examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"observed {len(examples)} examples; wrote {args.output}")


if __name__ == "__main__":
    main()
