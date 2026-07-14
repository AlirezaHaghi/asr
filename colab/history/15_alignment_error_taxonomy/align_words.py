"""Retrospective reconstruction; results require rerunning.

Create insertion-aware word alignments. This is not an original timestamped
development artifact.
"""

# خودمونی: خطای callsign و runway از یه کلمه معمولی مهم‌تره.

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def align(reference: str, hypothesis: str) -> list[dict]:
    ref, hyp = reference.split(), hypothesis.split()
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], back[i][0] = i, "deletion"
    for j in range(1, m + 1):
        cost[0][j], back[0][j] = j, "insertion"
    priority = {"substitution": 0, "deletion": 1, "insertion": 2}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost[i][j], back[i][j] = cost[i - 1][j - 1], "equal"
            else:
                choices = [(cost[i - 1][j - 1] + 1, "substitution"), (cost[i - 1][j] + 1, "deletion"), (cost[i][j - 1] + 1, "insertion")]
                value, operation = min(choices, key=lambda item: (item[0], priority[item[1]]))
                cost[i][j], back[i][j] = value, operation
    output, i, j = [], n, m
    while i or j:
        operation = back[i][j]
        if operation in ("equal", "substitution"):
            output.append({"operation": operation, "reference": ref[i - 1], "hypothesis": hyp[j - 1], "reference_index": i - 1, "hypothesis_index": j - 1})
            i, j = i - 1, j - 1
        elif operation == "deletion":
            output.append({"operation": operation, "reference": ref[i - 1], "hypothesis": None, "reference_index": i - 1, "hypothesis_index": None})
            i -= 1
        elif operation == "insertion":
            output.append({"operation": operation, "reference": None, "hypothesis": hyp[j - 1], "reference_index": None, "hypothesis_index": j - 1})
            j -= 1
        else:
            raise RuntimeError(f"Alignment backtrace failed at ref={i}, hyp={j}")
    return output[::-1]


def rows(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "predictions", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("Unsupported prediction schema")


def field(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        if row.get(key) is not None:
            return str(row[key]).lower().strip()
    raise KeyError(f"Missing field from {keys}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--reference")
    parser.add_argument("--hypothesis")
    parser.add_argument("--output", type=Path, default=Path("alignment.json"))
    args = parser.parse_args()
    try:
        aligned = []
        if args.input:
            for index, row in enumerate(rows(json.loads(args.input.read_text(encoding="utf-8")))):
                ref = field(row, ("reference", "reference_raw", "ref", "text"))
                hyp = field(row, ("hypothesis", "hypothesis_raw", "hyp", "prediction"))
                aligned.append({"id": row.get("id", index), "reference": ref, "hypothesis": hyp, "alignment": align(ref, hyp)})
        elif args.reference is not None and args.hypothesis is not None:
            aligned.append({"id": 0, "reference": args.reference, "hypothesis": args.hypothesis, "alignment": align(args.reference.lower(), args.hypothesis.lower())})
        else:
            parser.error("provide --input or both --reference and --hypothesis")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"records": aligned}, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
        return 0
    except Exception:
        logging.exception("word alignment failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
