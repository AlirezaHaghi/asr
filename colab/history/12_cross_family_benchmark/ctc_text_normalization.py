"""Retrospective reconstruction; results require a rerun.

Apply one transparent ATC scoring normalization to raw Whisper or CTC text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DIGITS = {str(i): word for i, word in enumerate("zero one two three four five six seven eight nine".split())}
REPLACEMENTS = [
    (r"\bniner\b", "nine"),
    (r"\btree\b", "three"),
    (r"\bfife\b", "five"),
    (r"\balfa\b", "alpha"),
    (r"\bryan\s+air\b", "ryanair"),
    (r"\bspeed\s+bird\b", "speedbird"),
    (r"\beuro\s+wings\b", "eurowings"),
]


def digit_words(value: str) -> str:
    return " ".join(DIGITS[digit] for digit in value)


def normalize(text: str) -> str:
    value = text.lower().strip()
    value = re.sub(r"\bfl\s*[- ]?(\d{2,3})\b", lambda m: "flight level " + digit_words(m.group(1)), value)
    value = re.sub(
        r"\brwy\s*[- ]?(\d{1,2})([lrc]?)\b",
        lambda m: "runway " + digit_words(m.group(1)) + ({"l": " left", "r": " right", "c": " center"}.get(m.group(2), "")),
        value,
    )
    value = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: digit_words(m.group(1)) + " decimal " + digit_words(m.group(2)), value)
    value = re.sub(r"\b\d+\b", lambda m: digit_words(m.group(0)), value)
    for pattern, replacement in REPLACEMENTS:
        value = re.sub(pattern, replacement, value)
    value = re.sub(r"[^a-z' ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", action="append", help="Text to normalize; repeat for multiple values")
    source.add_argument("--input", type=Path, help="JSONL input; stdin is used when neither option is supplied")
    parser.add_argument("--fields", nargs="+", default=["reference", "hypothesis"])
    parser.add_argument("--output", type=Path, help="Write text/JSONL here instead of stdout")
    return parser.parse_args()


def transform_jsonl(path: Path, fields: list[str]) -> list[str]:
    lines = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            for field in fields:
                if record.get(field) is not None:
                    record[f"{field}_normalized"] = normalize(str(record[field]))
            lines.append(json.dumps(record, ensure_ascii=False))
    return lines


def main() -> None:
    args = parse_args()
    if args.input:
        lines = transform_jsonl(args.input, args.fields)
    else:
        texts = args.text if args.text is not None else [line.rstrip("\n") for line in sys.stdin if line.strip()]
        lines = [normalize(text) for text in texts]
    rendered = "\n".join(lines) + ("\n" if lines else "")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {len(lines)} normalized records to {args.output}")
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
