"""Retrospective reconstruction from the final notebook/report; results require rerunning.

Audit how often ATC normalization patterns occur and alter a text corpus.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DIGITS = dict(zip("0123456789", "zero one two three four five six seven eight nine".split()))
RULES = {
    "icao_variant": r"\b(niner|alfa|tree|fife)\b",
    "flight_level_abbrev": r"\bfl\s*\d{2,3}\b",
    "runway_abbrev": r"\brwy\s*\d{1,2}[lrc]?\b",
    "digit_token": r"\b\d+\b",
    "airline_spacing": r"\b(ryan\s+air|euro\s+wings|sky\s+travel|speed\s+bird|bel\s+avia)\b",
}


def normalize(text: str) -> str:
    value = re.sub(r"[,\.!?;:\"()]", " ", text.lower().strip())
    replacements = [(r"\bniner\b","nine"),(r"\balfa\b","alpha"),(r"\btree\b","three"),
                    (r"\bfife\b","five"),(r"\bryan\s+air\b","ryanair"),
                    (r"\beuro\s+wings\b","eurowings"),(r"\bsky\s+travel\b","skytravel"),
                    (r"\bspeed\s+bird\b","speedbird"),(r"\bbel\s+avia\b","belavia"),
                    (r"\bairfrans\b","airfrance"),(r"\bok\b","okay")]
    for pattern, replacement in replacements: value = re.sub(pattern, replacement, value)
    words = lambda token: " ".join(DIGITS[d] for d in token)
    value = re.sub(r"\bfl\s*(\d{2,3})\b", lambda m: "flight level "+words(m.group(1)), value)
    def rwy(m: re.Match) -> str:
        side={"l":"left","r":"right","c":"center"}.get((m.group(2) or "").lower(),"")
        return " ".join(x for x in ("runway",words(m.group(1)),side) if x)
    value = re.sub(r"\brwy\s*(\d{1,2})([lrc]?)\b", rwy, value)
    value = re.sub(r"\b(\d+)\b", lambda m: words(m.group(1)), value)
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("audit.json"))
    args = parser.parse_args()
    lines = args.input.read_text(encoding="utf-8").splitlines()
    counts = Counter()
    changed_examples = []
    for line_number, line in enumerate(lines, 1):
        for name, pattern in RULES.items(): counts[name] += len(re.findall(pattern, line, re.I))
        after = normalize(line)
        if after != line.strip():
            counts["changed_lines"] += 1
            if len(changed_examples) < 25:
                changed_examples.append({"line": line_number, "before": line, "after": after})
    payload = {"source": str(args.input), "line_count": len(lines),
               "counts": dict(counts), "changed_examples": changed_examples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"audited {len(lines)} lines; {counts['changed_lines']} changed")


if __name__ == "__main__":
    main()
