"""Retrospective reconstruction from the final notebook/report; results require rerunning.

Format spoken ATC flight levels, runways, and frequencies for display.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

WORD_TO_DIGIT = dict(zip("zero one two three four five six seven eight nine".split(), "0123456789"))
WORD_TO_DIGIT["niner"] = "9"
NUMBER_PATTERN = "|".join(map(re.escape, WORD_TO_DIGIT))


def digits(words: str) -> str:
    return "".join(WORD_TO_DIGIT[word.lower()] for word in words.split() if word.lower() in WORD_TO_DIGIT)


def format_display(text: str) -> str:
    def flight_level(match: re.Match) -> str:
        value = digits(match.group(1))
        return "FL" + value if value else match.group(0)
    value = re.sub(rf"\bflight level\s+((?:{NUMBER_PATTERN})(?:\s+(?:{NUMBER_PATTERN}))*)",
                   flight_level, text, flags=re.I)
    def runway(match: re.Match) -> str:
        value = digits(match.group(1))
        side = {"left":"L", "right":"R", "center":"C"}.get((match.group(2) or "").lower(), "")
        return "RWY" + value + side if value else match.group(0)
    value = re.sub(rf"\brunway\s+((?:{NUMBER_PATTERN})(?:\s+(?:{NUMBER_PATTERN}))*)"
                   rf"\s*(left|right|center)?", runway, value, flags=re.I)
    def frequency(match: re.Match) -> str:
        return digits(match.group(1)) + "." + digits(match.group(2))
    value = re.sub(rf"((?:{NUMBER_PATTERN})(?:\s+(?:{NUMBER_PATTERN}))*)\s+decimal\s+"
                   rf"((?:{NUMBER_PATTERN})(?:\s+(?:{NUMBER_PATTERN}))*)", frequency, value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lines = [args.text] if args.text is not None else args.input.read_text(encoding="utf-8").splitlines()
    rendered = "\n".join(format_display(line) for line in lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + ("\n" if lines else ""), encoding="utf-8")
        print(f"formatted {len(lines)} lines to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
