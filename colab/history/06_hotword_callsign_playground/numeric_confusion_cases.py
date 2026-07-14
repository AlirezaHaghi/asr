"""Retrospective reconstruction; rerun to generate ATC numeric stress cases."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


DIGITS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "niner",
}


def spoken(number: str) -> str:
    return " ".join(DIGITS[digit] for digit in number)


def cases():
    for heading in ("060", "160", "260", "360"):
        yield {"kind": "heading", "written": heading, "text": f"turn right heading {spoken(heading)}"}
    for level in ("080", "180", "280", "380"):
        yield {"kind": "flight_level", "written": f"FL{level}", "text": f"climb flight level {spoken(level)}"}
    for runway, side in itertools.product(("06", "16", "26", "36"), ("left", "right")):
        yield {"kind": "runway", "written": f"RWY{runway}{side[0].upper()}", "text": f"runway {spoken(runway)} {side}"}
    for frequency in ("118.60", "119.60", "128.60", "129.60"):
        left, right = frequency.split(".")
        yield {"kind": "frequency", "written": frequency, "text": f"contact {spoken(left)} decimal {spoken(right)}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = list(cases())
    if args.output:
        args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        print(f"wrote {len(rows)} stress cases to {args.output}")
    else:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
