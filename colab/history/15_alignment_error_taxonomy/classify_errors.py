"""Retrospective reconstruction; results require rerunning.

Classify insertion-aware ATC word errors from supplied predictions. This is
not an original timestamped development artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import re
from pathlib import Path


NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "niner", "ten", "hundred", "thousand", "decimal"}
COMMAND_WORDS = {"climb", "descend", "cleared", "contact", "hold", "maintain", "turn", "heading", "runway", "taxi", "land", "takeoff", "squawk", "frequency", "level"}


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None, str | None]]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = i, "deletion"
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = j, "insertion"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j], back[i][j] = dp[i - 1][j - 1], "equal"
            else:
                value, _, operation = min((dp[i - 1][j - 1] + 1, 0, "substitution"), (dp[i - 1][j] + 1, 1, "deletion"), (dp[i][j - 1] + 1, 2, "insertion"))
                dp[i][j], back[i][j] = value, operation
    result, i, j = [], n, m
    while i or j:
        operation = back[i][j]
        if operation in ("equal", "substitution"):
            result.append((operation, ref[i - 1], hyp[j - 1])); i, j = i - 1, j - 1
        elif operation == "deletion":
            result.append((operation, ref[i - 1], None)); i -= 1
        elif operation == "insertion":
            result.append((operation, None, hyp[j - 1])); j -= 1
        else:
            raise RuntimeError("broken alignment backtrace")
    return result[::-1]


def category(reference: str | None, hypothesis: str | None) -> str:
    tokens = {token for token in (reference, hypothesis) if token}
    if tokens & NUMBER_WORDS or any(re.fullmatch(r"\d+(?:\.\d+)?", token) for token in tokens):
        return "number"
    if tokens & COMMAND_WORDS:
        return "command"
    if any(any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token) for token in tokens):
        return "callsign"
    return "other"


def severity(kind: str) -> str:
    return "moderate" if kind in {"number", "command", "callsign"} else "minor"


def get_rows(payload: object) -> list[dict]:
    if isinstance(payload, list): return payload
    if isinstance(payload, dict):
        for key in ("records", "predictions", "items"):
            if isinstance(payload.get(key), list): return payload[key]
    raise ValueError("Unsupported prediction JSON")


def get(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        if row.get(key) is not None: return str(row[key]).lower().strip()
    raise KeyError(f"Missing one of {keys}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("error_taxonomy.json"))
    args = parser.parse_args()
    try:
        errors, operation_counts, category_counts, severity_counts = [], Counter(), Counter(), Counter()
        for index, row in enumerate(get_rows(json.loads(args.predictions.read_text(encoding="utf-8")))):
            ref = get(row, ("reference", "reference_raw", "ref", "text"))
            hyp = get(row, ("hypothesis", "hypothesis_raw", "hyp", "prediction"))
            for position, (operation, left, right) in enumerate(align(ref.split(), hyp.split())):
                if operation == "equal": continue
                kind = category(left, right); level = severity(kind)
                operation_counts[operation] += 1; category_counts[kind] += 1; severity_counts[level] += 1
                errors.append({"record_id": row.get("id", index), "alignment_position": position, "operation": operation, "reference": left, "hypothesis": right, "category": kind, "severity": level})
        payload = {"source": str(args.predictions), "rules": {"severity": "number/command/callsign => moderate; other => minor", "note": "heuristic review aid, not a safety assessment"}, "counts": {"operation": operation_counts, "category": category_counts, "severity": severity_counts}, "errors": errors}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output} ({len(errors)} errors)")
        return 0
    except Exception:
        logging.exception("error classification failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
