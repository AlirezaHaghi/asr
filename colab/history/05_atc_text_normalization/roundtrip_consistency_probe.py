"""Retrospective reconstruction from the final notebook/report; results require rerunning.

Probe spoken-to-display-to-spoken consistency for bounded ATC templates.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DIGIT_WORDS = dict(zip("0123456789", "zero one two three four five six seven eight nine".split()))
WORDS = {word: digit for digit, word in DIGIT_WORDS.items()}
WORDS["niner"] = "9"
P = "|".join(map(re.escape, WORDS))


def display(text: str) -> str:
    number=lambda s:"".join(WORDS[w.lower()] for w in s.split())
    text=re.sub(rf"\bflight level\s+((?:{P})(?:\s+(?:{P}))*)",lambda m:"FL"+number(m.group(1)),text,flags=re.I)
    def runway(m: re.Match) -> str:
        side={"left":"L","right":"R","center":"C"}.get((m.group(2) or "").lower(),"")
        return "RWY"+number(m.group(1))+side
    text=re.sub(rf"\brunway\s+((?:{P})(?:\s+(?:{P}))*)\s*(left|right|center)?",runway,text,flags=re.I)
    text=re.sub(rf"((?:{P})(?:\s+(?:{P}))*)\s+decimal\s+((?:{P})(?:\s+(?:{P}))*)",
                lambda m:number(m.group(1))+"."+number(m.group(2)),text,flags=re.I)
    return re.sub(r"\s+"," ",text).strip()


def expand(value: str) -> str:
    words=lambda s:" ".join(DIGIT_WORDS[d] for d in s)
    value=re.sub(r"\bFL(\d{2,3})\b",lambda m:"flight level "+words(m.group(1)),value,flags=re.I)
    def runway(m: re.Match) -> str:
        side={"L":"left","R":"right","C":"center"}.get((m.group(2) or "").upper(),"")
        return " ".join(x for x in ("runway",words(m.group(1)),side) if x)
    value=re.sub(r"\bRWY(\d{1,2})([LRC]?)\b",runway,value,flags=re.I)
    value=re.sub(r"\b(\d+)\.(\d+)\b",lambda m:words(m.group(1))+" decimal "+words(m.group(2)),value)
    return re.sub(r"\s+"," ",value).strip().lower()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("roundtrip.json"))
    args=parser.parse_args()
    cases=["descend flight level one eight zero","cleared runway two eight left",
           "contact one two seven decimal four","line up runway zero nine center"]
    rows=[]
    for source in cases:
        shown=display(source); recovered=expand(shown)
        rows.append({"source":source,"display":shown,"expanded":recovered,"consistent":source==recovered})
    payload={"cases":rows,"consistent_count":sum(row["consistent"] for row in rows),"total":len(rows)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))


if __name__ == "__main__":
    main()
