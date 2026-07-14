"""Retrospective reconstruction; results require rerunning.

Measure normalization rule-family effects on supplied predictions. This is not
an original timestamped development artifact.
"""

# خودمونی: هر بار فقط یه rule رو بردار تا اثرش معلوم باشه.

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path


DIGITS = {str(index): word for index, word in enumerate("zero one two three four five six seven eight nine".split())}


def apply(text: str, profile: str) -> str:
    value = re.sub(r"\s+", " ", text.lower()).strip()
    if profile == "raw": return value
    value = re.sub(r"[,\.!?;:\"()\[\]{}]", " ", value)
    if profile == "punctuation": return re.sub(r"\s+", " ", value).strip()
    for source, target in (("niner", "nine"), ("alfa", "alpha"), ("tree", "three"), ("fife", "five")):
        value = re.sub(rf"\b{source}\b", target, value)
    if profile == "icao": return re.sub(r"\s+", " ", value).strip()
    words = lambda digits: " ".join(DIGITS[character] for character in digits)
    value = re.sub(r"\bfl\s*(\d{2,3})\b", lambda match: "flight level " + words(match.group(1)), value)
    value = re.sub(r"\b(?:rwy|runway)\s*(\d{1,2})([lrc]?)\b", lambda match: "runway " + words(match.group(1)) + {"l": " left", "r": " right", "c": " center", "": ""}[match.group(2).lower()], value)
    value = re.sub(r"\b(\d+)\.(\d+)\b", lambda match: words(match.group(1)) + " decimal " + words(match.group(2)), value)
    value = re.sub(r"\b\d+\b", lambda match: words(match.group(0)), value)
    return re.sub(r"\s+", " ", value).strip()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, left in enumerate(ref, 1):
        current = [i]
        for j, right in enumerate(hyp, 1):
            current.append(previous[j - 1] if left == right else 1 + min(previous[j - 1], previous[j], current[-1]))
        previous = current
    return previous[-1]


def rows(payload: object) -> list[dict]:
    if isinstance(payload, list): return payload
    if isinstance(payload, dict):
        for key in ("records", "predictions", "items"):
            if isinstance(payload.get(key), list): return payload[key]
    raise ValueError("Unsupported predictions JSON")


def text(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        if row.get(key) is not None: return str(row[key])
    raise KeyError(f"Missing one of {keys}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("normalization_ablation.json"))
    args = parser.parse_args()
    try:
        data = rows(json.loads(args.predictions.read_text(encoding="utf-8")))
        results = []
        for profile in ("raw", "punctuation", "icao", "atc"):
            words = errors = 0
            for row in data:
                ref = apply(text(row, ("reference", "reference_raw", "ref", "text")), profile).split()
                hyp = apply(text(row, ("hypothesis", "hypothesis_raw", "hyp", "prediction")), profile).split()
                words += len(ref); errors += edit_distance(ref, hyp)
            results.append({"profile": profile, "reference_words": words, "errors": errors, "wer": errors / words if words else None})
        payload = {"source": str(args.predictions), "samples": len(data), "profiles": results, "note": "all metrics were recomputed from this source"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
        return 0
    except Exception:
        logging.exception("normalization ablation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
