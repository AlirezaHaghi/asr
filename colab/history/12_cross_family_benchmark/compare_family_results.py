"""Retrospective reconstruction; rerun metrics on completed cross-family JSONL."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["source_file"] = str(path)
                rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    from jiwer import cer, process_words

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("error"):
            grouped[row["run_id"]].append(row)
    output = []
    for run_id, records in grouped.items():
        refs = [str(row.get("reference", "")) for row in records]
        hyps = [str(row.get("hypothesis", "")) for row in records]
        alignment = process_words(refs, hyps)
        duration = sum(float(row.get("duration_s") or 0) for row in records)
        inference = sum(float(row.get("inference_s") or 0) for row in records)
        output.append({
            "run_id": run_id,
            "model_id": records[0].get("model_id"),
            "family": records[0].get("family"),
            "samples": len(records),
            "wer": alignment.wer,
            "cer": cer(refs, hyps),
            "substitutions": alignment.substitutions,
            "deletions": alignment.deletions,
            "insertions": alignment.insertions,
            "rtf": inference / duration if duration else None,
        })
    return sorted(output, key=lambda row: row["wer"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(load(args.predictions))
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
