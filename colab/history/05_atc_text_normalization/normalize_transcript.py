"""Retrospective reconstruction from the final notebook/report; results require rerunning.

Normalize ATC text to the comparison form used for WER scoring.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

DIGITS = dict(zip("0123456789", "zero one two three four five six seven eight nine".split()))
REPLACEMENTS = [
    (r"\bniner\b", "nine"), (r"\balfa\b", "alpha"), (r"\btree\b", "three"),
    (r"\bfife\b", "five"), (r"\bryan\s+air\b", "ryanair"),
    (r"\beuro\s+wings\b", "eurowings"), (r"\bsky\s+travel\b", "skytravel"),
    (r"\bspeed\s+bird\b", "speedbird"), (r"\bbel\s+avia\b", "belavia"),
    (r"\bairfrans\b", "airfrance"), (r"\bok\b", "okay"),
]


def digit_words(token: str) -> str:
    return " ".join(DIGITS[character] for character in token)


def normalize(text: str) -> str:
    value = re.sub(r"[,\.!?;:\"()]", " ", text.lower().strip())
    for pattern, replacement in REPLACEMENTS:
        value = re.sub(pattern, replacement, value)
    value = re.sub(r"\bfl\s*(\d{2,3})\b", lambda m: "flight level " + digit_words(m.group(1)), value)
    def runway(match: re.Match) -> str:
        side = {"l":"left", "r":"right", "c":"center"}.get((match.group(2) or "").lower(), "")
        return " ".join(part for part in ("runway", digit_words(match.group(1)), side) if part)
    value = re.sub(r"\brwy\s*(\d{1,2})([lrc]?)\b", runway, value)
    value = re.sub(r"\b(\d+)\b", lambda m: digit_words(m.group(1)), value)
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lines = [args.text] if args.text is not None else args.input.read_text(encoding="utf-8").splitlines()
    rendered = "\n".join(normalize(line) for line in lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + ("\n" if lines else ""), encoding="utf-8")
        print(f"normalized {len(lines)} lines to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
