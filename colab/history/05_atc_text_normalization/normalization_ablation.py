"""Retrospective reconstruction from the final notebook/report; results require rerunning.

Measure WER while progressively enabling ATC normalization rule families.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from jiwer import wer

DIGITS = dict(zip("0123456789", "zero one two three four five six seven eight nine".split()))
ICAO = [(r"\bniner\b","nine"),(r"\balfa\b","alpha"),(r"\btree\b","three"),(r"\bfife\b","five")]
AIRLINES = [(r"\bryan\s+air\b","ryanair"),(r"\beuro\s+wings\b","eurowings"),
            (r"\bsky\s+travel\b","skytravel"),(r"\bspeed\s+bird\b","speedbird"),
            (r"\bbel\s+avia\b","belavia"),(r"\bairfrans\b","airfrance")]


def transform(text: str, stage: str) -> str:
    value = text.lower().strip()
    if stage == "raw_lower":
        return value
    value = re.sub(r"[,\.!?;:\"()]", " ", value)
    if stage in {"icao", "airlines", "numeric", "full"}:
        for pattern, replacement in ICAO: value = re.sub(pattern, replacement, value)
    if stage in {"airlines", "numeric", "full"}:
        for pattern, replacement in AIRLINES: value = re.sub(pattern, replacement, value)
    if stage in {"numeric", "full"}:
        words = lambda token: " ".join(DIGITS[d] for d in token)
        value = re.sub(r"\bfl\s*(\d{2,3})\b", lambda m: "flight level "+words(m.group(1)), value)
        def rwy(m: re.Match) -> str:
            side={"l":"left","r":"right","c":"center"}.get((m.group(2) or "").lower(),"")
            return " ".join(x for x in ("runway",words(m.group(1)),side) if x)
        value = re.sub(r"\brwy\s*(\d{1,2})([lrc]?)\b", rwy, value)
        value = re.sub(r"\b(\d+)\b", lambda m: words(m.group(1)), value)
    if stage == "full":
        value = re.sub(r"\bok\b", "okay", value)
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True,
                        help="Rows with reference and hypothesis strings")
    parser.add_argument("--output", type=Path, default=Path("ablation.json"))
    args = parser.parse_args()
    pairs = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    stages = ["raw_lower", "punctuation", "icao", "airlines", "numeric", "full"]
    results = []
    for stage in stages:
        refs = [transform(row["reference"], stage) for row in pairs]
        hyps = [transform(row["hypothesis"], stage) for row in pairs]
        results.append({"stage": stage, "wer": wer(refs, hyps), "rows": len(pairs)})
    payload = {"source": str(args.jsonl), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in results: print(f"{row['stage']:<12} WER={row['wer']:.4f}")


if __name__ == "__main__":
    main()
