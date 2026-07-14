"""Retrospective reconstruction from the final notebook/report; results require a rerun.

Inspect local Hugging Face cache state without downloading artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_IDS = [
    "Jzuluaga/atco2_corpus_1h",
    "jacktol/whisper-large-v3-finetuned-for-ATC",
    "jacktol/whisper-medium.en-fine-tuned-for-ATC",
    "fjmgAI/whisper-large-v3-ATC",
    "Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim",
]


def directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*") if path.exists() else ():
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    parser.add_argument("--cache-dir", type=Path, default=default_cache)
    parser.add_argument("--output", type=Path, default=Path("hf_cache.json"))
    args = parser.parse_args()
    entries = []
    if args.cache_dir.exists():
        for child in sorted(args.cache_dir.iterdir()):
            entries.append({"name": child.name, "path": str(child), "bytes": directory_size(child)})
    payload = {
        "cache_dir": str(args.cache_dir.resolve()),
        "exists": args.cache_dir.exists(),
        "entries": entries,
        "project_identifiers": PROJECT_IDS,
        "note": "Presence in this inventory is not a model integrity check.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"found {len(entries)} top-level cache entries; wrote {args.output}")


if __name__ == "__main__":
    main()
