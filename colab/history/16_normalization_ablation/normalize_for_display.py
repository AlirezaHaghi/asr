"""Retrospective reconstruction; results require rerunning.

Format spoken ATC number sequences for display. This is not an original
timestamped development artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path


WORDS = {word: str(index) for index, word in enumerate("zero one two three four five six seven eight nine".split())}
WORDS["niner"] = "9"
NUMBER = "|".join(WORDS)


def digits(words: str) -> str:
    return "".join(WORDS[word.lower()] for word in words.split())


def normalize(text: str) -> str:
    value = text.strip()
    value = re.sub(rf"\bflight level\s+((?:{NUMBER})(?:\s+(?:{NUMBER}))*)", lambda match: "FL" + digits(match.group(1)), value, flags=re.IGNORECASE)
    def runway(match: re.Match[str]) -> str:
        side = {"left": "L", "right": "R", "center": "C"}.get((match.group(2) or "").lower(), "")
        return "RWY" + digits(match.group(1)) + side
    value = re.sub(rf"\brunway\s+((?:{NUMBER})(?:\s+(?:{NUMBER}))*)\s*(left|right|center)?", runway, value, flags=re.IGNORECASE)
    value = re.sub(rf"((?:{NUMBER})(?:\s+(?:{NUMBER}))*)\s+decimal\s+((?:{NUMBER})(?:\s+(?:{NUMBER}))*)", lambda match: digits(match.group(1)) + "." + digits(match.group(2)), value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("display_normalized.json"))
    args = parser.parse_args()
    try:
        texts = [args.text] if args.text is not None else args.input.read_text(encoding="utf-8").splitlines()
        output = [{"before": item, "after": normalize(item)} for item in texts]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.output}")
        return 0
    except Exception:
        logging.exception("display normalization failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
