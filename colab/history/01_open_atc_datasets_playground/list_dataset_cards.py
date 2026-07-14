"""Retrospective reconstruction; rerun to fetch current ATC dataset-card metadata.

Without ``--online`` this only prints the local, carefully qualified catalog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CATALOG = Path(__file__).with_name("dataset_catalog.json")


def load_catalog(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("catalog must contain a JSON list")
    return rows


def fetch_card(dataset_id: str) -> dict:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(dataset_id)
    card = getattr(info, "card_data", None)
    return {
        "id": info.id,
        "sha": info.sha,
        "last_modified": str(info.last_modified),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "tags": list(getattr(info, "tags", None) or []),
        "card_data": dict(card) if card else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--online", action="store_true", help="query Hugging Face now")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = load_catalog(args.catalog)
    for row in rows:
        dataset_id = row.get("hf_id")
        if not args.online or not dataset_id:
            continue
        try:
            row["live_card"] = fetch_card(dataset_id)
        except Exception as exc:  # اینجا خطا رو قایم نمی‌کنیم؛ می‌ذاریم توی گزارش
            row["live_card_error"] = f"{type(exc).__name__}: {exc}"

    text = json.dumps(rows, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
