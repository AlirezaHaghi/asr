"""Retrospective reconstruction; results require rerunning.

Compare supplied VAD runs. This is not an original timestamped development
artifact.
"""

# خودمونی: segment بیشتر همیشه یعنی VAD بهتر نیست.

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


REF_KEYS = ("reference", "reference_raw", "ref", "text")
HYP_KEYS = ("hypothesis", "hypothesis_raw", "hyp", "prediction")


def records(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "predictions", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("Expected a JSON list or an object containing records/predictions/items")


def text_from(record: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in record:
            return str(record[key]).strip().lower()
    raise KeyError(f"Record is missing all fields: {keys}")


def edit_counts(reference: str, hypothesis: str) -> dict[str, int]:
    ref, hyp = reference.split(), hypothesis.split()
    table = [[(0, 0, 0, 0) for _ in range(len(hyp) + 1)] for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, len(hyp) + 1):
        table[0][j] = (j, 0, 0, j)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance, s, d, ins = table[i - 1][j - 1]
                table[i][j] = (distance, s, d, ins)
            else:
                sub = table[i - 1][j - 1]
                delete = table[i - 1][j]
                insert = table[i][j - 1]
                candidates = [(sub[0] + 1, sub[1] + 1, sub[2], sub[3]), (delete[0] + 1, delete[1], delete[2] + 1, delete[3]), (insert[0] + 1, insert[1], insert[2], insert[3] + 1)]
                table[i][j] = min(candidates, key=lambda value: (value[0], value[3], value[2], value[1]))
    distance, substitutions, deletions, insertions = table[-1][-1]
    return {"reference_words": len(ref), "errors": distance, "substitutions": substitutions, "deletions": deletions, "insertions": insertions}


def summarize(path: Path) -> dict:
    rows = records(json.loads(path.read_text(encoding="utf-8")))
    totals = {"reference_words": 0, "errors": 0, "substitutions": 0, "deletions": 0, "insertions": 0}
    for row in rows:
        counts = edit_counts(text_from(row, REF_KEYS), text_from(row, HYP_KEYS))
        for key in totals:
            totals[key] += counts[key]
    totals["samples"] = len(rows)
    totals["wer"] = totals["errors"] / totals["reference_words"] if totals["reference_words"] else None
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("vad", type=Path)
    parser.add_argument("--output", type=Path, default=Path("vad_comparison.json"))
    args = parser.parse_args()
    try:
        baseline = summarize(args.baseline)
        vad = summarize(args.vad)
        if baseline["samples"] != vad["samples"]:
            raise ValueError("The two files contain different sample counts")
        delta = None if baseline["wer"] is None or vad["wer"] is None else vad["wer"] - baseline["wer"]
        result = {"baseline_file": str(args.baseline), "vad_file": str(args.vad), "baseline": baseline, "vad": vad, "vad_minus_baseline_wer": delta}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
        return 0
    except Exception:
        logging.exception("VAD run comparison failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
