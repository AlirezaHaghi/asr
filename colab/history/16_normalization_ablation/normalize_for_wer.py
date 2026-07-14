"""Retrospective reconstruction; results require rerunning.

Apply transparent ATC scoring normalization. This is not an original timestamped
development artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path


DIGITS = {str(index): word for index, word in enumerate("zero one two three four five six seven eight nine".split())}
AIRLINES = ((r"\bryan\s+air\b", "ryanair"), (r"\bspeed\s+bird\b", "speedbird"), (r"\beuro\s+wings\b", "eurowings"), (r"\bair\s+france\b", "airfrance"))


def digit_words(value: str) -> str:
    return " ".join(DIGITS[character] for character in value)


def normalize(text: str) -> str:
    value = text.lower().strip()
    value = re.sub(r"[,\.!?;:\"()\[\]{}]", " ", value)
    for source, target in ((r"\bniner\b", "nine"), (r"\balfa\b", "alpha"), (r"\btree\b", "three"), (r"\bfife\b", "five")):
        value = re.sub(source, target, value)
    for source, target in AIRLINES:
        value = re.sub(source, target, value)
    value = re.sub(r"\bfl\s*(\d{2,3})\b", lambda match: "flight level " + digit_words(match.group(1)), value)
    def runway(match: re.Match[str]) -> str:
        side = {"l": "left", "r": "right", "c": "center"}.get((match.group(2) or "").lower(), "")
        return ("runway " + digit_words(match.group(1)) + (" " + side if side else "")).strip()
    value = re.sub(r"\b(?:rwy|runway)\s*(\d{1,2})([lrc]?)\b", runway, value)
    value = re.sub(r"\b(\d+)\.(\d+)\b", lambda match: digit_words(match.group(1)) + " decimal " + digit_words(match.group(2)), value)
    value = re.sub(r"\b\d+\b", lambda match: digit_words(match.group(0)), value)
    return re.sub(r"\s+", " ", value).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input", type=Path)
    parser.add_argument("--json-field", help="normalize this field in each object of a JSON list")
    parser.add_argument("--output", type=Path, default=Path("wer_normalized.json"))
    args = parser.parse_args()
    try:
        if args.text is not None:
            output: object = {"before": args.text, "after": normalize(args.text)}
        elif args.json_field:
            payload = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(payload, list): raise ValueError("JSON mode expects a list")
            output = [{"before": row.get(args.json_field), "after": normalize(str(row.get(args.json_field, "")))} for row in payload]
        else:
            output = [{"before": line, "after": normalize(line)} for line in args.input.read_text(encoding="utf-8").splitlines()]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.output}")
        return 0
    except Exception:
        logging.exception("WER normalization failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
