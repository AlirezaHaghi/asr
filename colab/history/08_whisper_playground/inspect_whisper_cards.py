"""Retrospective reconstruction; rerun to inspect live Whisper model metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CATALOG = Path(__file__).with_name("whisper_model_catalog.json")


def live_info(model_id: str) -> dict:
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id)
    card = getattr(info, "card_data", None)
    return {
        "sha": info.sha,
        "last_modified": str(info.last_modified),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "library_name": getattr(info, "library_name", None),
        "tags": list(getattr(info, "tags", None) or []),
        "card_data": dict(card) if card else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.catalog.read_text(encoding="utf-8"))

    if args.online:
        for row in rows:
            try:
                row["live"] = live_info(row["model_id"])
            except Exception as exc:  # نت قطع بود، کل کار رو هوا نمی‌کنیم
                row["live_error"] = f"{type(exc).__name__}: {exc}"

    text = json.dumps(rows, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
