"""Retrospective reconstruction; rerun to create a cross-family benchmark plan."""

# خودمونی: loaderها فرق دارن، ولی schema نتیجه باید یکی باشه.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CATALOG = Path(__file__).with_name("cross_family_catalog.json")


def fingerprint(row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--dataset", default="Jzuluaga/atco2_corpus_1h")
    parser.add_argument("--split", default="test")
    parser.add_argument("--normalizer", default="atc_wer_v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for model in json.loads(args.catalog.read_text(encoding="utf-8")):
        row = {
            **model,
            "dataset": args.dataset,
            "split": args.split,
            "limit": args.limit,
            "normalizer": args.normalizer,
            "status": "planned_not_executed",
        }
        row["run_id"] = fingerprint(row)
        rows.append(row)
    report = {"caveat": "same manifest and normalizer required", "runs": rows}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
