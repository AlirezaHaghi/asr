"""Retrospective reconstruction; results require rerunning.

Validate metric edge cases without asserting project benchmark results. This
is not an original timestamped development artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def operations(reference: list[str], hypothesis: list[str]) -> list[str]:
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = i, "D"
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j], back[i][j] = dp[i - 1][j - 1], "="
            else:
                value, _, code = min((dp[i - 1][j - 1] + 1, 0, "S"), (dp[i - 1][j] + 1, 1, "D"), (dp[i][j - 1] + 1, 2, "I"))
                dp[i][j], back[i][j] = value, code
    result, i, j = [], n, m
    while i or j:
        code = back[i][j]
        result.append(code)
        if code in ("=", "S"):
            i, j = i - 1, j - 1
        elif code == "D":
            i -= 1
        elif code == "I":
            j -= 1
        else:
            raise RuntimeError("invalid backtrace")
    return result[::-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("metric_validation.json"))
    args = parser.parse_args()
    try:
        cases = [
            ("middle insertion", ["alpha", "bravo"], ["alpha", "extra", "bravo"], ["=", "I", "="]),
            ("deletion", ["climb", "flight", "level"], ["climb", "level"], ["=", "D", "="]),
            ("substitution", ["runway", "two"], ["runway", "three"], ["=", "S"]),
            ("leading insertion", ["contact"], ["please", "contact"], ["I", "="]),
        ]
        checks = []
        for name, ref, hyp, expected in cases:
            actual = operations(ref, hyp)
            if actual != expected:
                raise AssertionError(f"{name}: expected {expected}, got {actual}")
            checks.append({"case": name, "passed": True, "operations": actual})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"checks": checks}, indent=2), encoding="utf-8")
        print(f"validated {len(checks)} alignment cases")
        return 0
    except Exception:
        logging.exception("metric validation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
