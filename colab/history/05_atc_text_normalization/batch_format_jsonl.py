"""Retrospective reconstruction from the final notebook/report; results require rerunning.

Add human-facing ATC display text to prediction JSONL records.
"""

# خودمونی: normalizer متن رو مرتب می‌کنه، جواب رو حدس نمی‌زنه.
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WORDS = dict(zip("zero one two three four five six seven eight nine".split(), "0123456789"))
WORDS["niner"] = "9"
PATTERN = "|".join(map(re.escape, WORDS))


def make_digits(value: str) -> str:
    return "".join(WORDS[word.lower()] for word in value.split() if word.lower() in WORDS)


def display(text: str) -> str:
    def fl(m: re.Match) -> str: return "FL" + make_digits(m.group(1))
    text = re.sub(rf"\bflight level\s+((?:{PATTERN})(?:\s+(?:{PATTERN}))*)", fl, text, flags=re.I)
    def rwy(m: re.Match) -> str:
        side={"left":"L","right":"R","center":"C"}.get((m.group(2) or "").lower(),"")
        return "RWY"+make_digits(m.group(1))+side
    text = re.sub(rf"\brunway\s+((?:{PATTERN})(?:\s+(?:{PATTERN}))*)\s*(left|right|center)?",
                  rwy, text, flags=re.I)
    def freq(m: re.Match) -> str: return make_digits(m.group(1))+"."+make_digits(m.group(2))
    text = re.sub(rf"((?:{PATTERN})(?:\s+(?:{PATTERN}))*)\s+decimal\s+"
                  rf"((?:{PATTERN})(?:\s+(?:{PATTERN}))*)", freq, text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--field", default="hypothesis_raw")
    parser.add_argument("--output-field", default="hypothesis_display")
    parser.add_argument("--summary", type=Path, default=Path("display_summary.json"))
    args = parser.parse_args()
    rows, changed, missing = [], 0, 0
    for number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        row = json.loads(line)
        if args.field not in row:
            missing += 1; row[args.output_field] = None
        else:
            row[args.output_field] = display(str(row[args.field]))
            changed += row[args.output_field] != row[args.field]
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False)+"\n")
    summary={"input":str(args.input),"rows":len(rows),"changed":changed,"missing_field":missing,
             "source_field":args.field,"output_field":args.output_field}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
