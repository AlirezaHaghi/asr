"""Retrospective reconstruction; results require rerunning.

Compute insertion-aware WER, CER, and S/D/I from supplied predictions. This is
not an original timestamped development artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence


REF_KEYS = ("reference", "reference_raw", "ref", "text")
HYP_KEYS = ("hypothesis", "hypothesis_raw", "hyp", "prediction")


def unpack(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "predictions", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Prediction JSON must be a list or contain records/predictions/items")


def get_text(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    raise KeyError(f"Missing one of {keys}")


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> list[dict]:
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    back: list[list[str | None]] = [[None] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0], back[i][0] = i, "delete"
    for j in range(1, cols):
        cost[0][j], back[0][j] = j, "insert"
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j], back[i][j] = cost[i - 1][j - 1], "equal"
            else:
                candidates = [(cost[i - 1][j - 1] + 1, 0, "substitute"), (cost[i - 1][j] + 1, 1, "delete"), (cost[i][j - 1] + 1, 2, "insert")]
                best = min(candidates)
                cost[i][j], back[i][j] = best[0], best[2]
    operations: list[dict] = []
    i, j = len(reference), len(hypothesis)
    while i or j:
        op = back[i][j]
        if op in ("equal", "substitute"):
            operations.append({"operation": op, "reference": reference[i - 1], "hypothesis": hypothesis[j - 1]})
            i, j = i - 1, j - 1
        elif op == "delete":
            operations.append({"operation": op, "reference": reference[i - 1], "hypothesis": None})
            i -= 1
        elif op == "insert":
            operations.append({"operation": op, "reference": None, "hypothesis": hypothesis[j - 1]})
            j -= 1
        else:
            raise RuntimeError(f"Broken backtrace at {i}, {j}")
    return list(reversed(operations))


def counts(operations: list[dict], reference_length: int) -> dict:
    substitutions = sum(op["operation"] == "substitute" for op in operations)
    deletions = sum(op["operation"] == "delete" for op in operations)
    insertions = sum(op["operation"] == "insert" for op in operations)
    errors = substitutions + deletions + insertions
    return {"reference_units": reference_length, "errors": errors, "substitutions": substitutions, "deletions": deletions, "insertions": insertions, "error_rate": errors / reference_length if reference_length else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    args = parser.parse_args()
    try:
        rows = unpack(json.loads(args.predictions.read_text(encoding="utf-8")))
        word_totals = {key: 0 for key in ("reference_units", "errors", "substitutions", "deletions", "insertions")}
        char_totals = word_totals.copy()
        details = []
        for index, row in enumerate(rows):
            ref, hyp = get_text(row, REF_KEYS), get_text(row, HYP_KEYS)
            word = counts(align(ref.split(), hyp.split()), len(ref.split()))
            ref_chars, hyp_chars = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
            char = counts(align(ref_chars, hyp_chars), len(ref_chars))
            for key in word_totals:
                word_totals[key] += word[key]
                char_totals[key] += char[key]
            details.append({"id": row.get("id", index), "word": word, "character": char, "exact": ref == hyp})
        if word_totals["reference_units"] == 0:
            raise ValueError("Corpus contains no reference words")
        word_totals["error_rate"] = word_totals["errors"] / word_totals["reference_units"]
        char_totals["error_rate"] = char_totals["errors"] / char_totals["reference_units"] if char_totals["reference_units"] else None
        payload = {"source": str(args.predictions), "samples": len(rows), "perfect": sum(item["exact"] for item in details), "word": word_totals, "character": char_totals, "records": details}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
        return 0
    except Exception:
        logging.exception("metric computation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
