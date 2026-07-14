"""Retrospective reconstruction; rerun to print an RNNT-versus-CTC test plan."""

from __future__ import annotations

import argparse
import json


def plan(beam_sizes: list[int]) -> list[dict]:
    rows = [{"decoder": "rnnt_greedy", "beam": 1, "note": "fast baseline"}]
    rows.extend({"decoder": "rnnt_beam", "beam": beam, "note": "verify NeMo model support"} for beam in beam_sizes)
    rows.append({"decoder": "ctc_greedy", "beam": 1, "note": "hybrid head if checkpoint exposes it"})
    rows.append({"decoder": "ctc_lexicon", "beam": 100, "note": "optional ATC LM/hotwords"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-sizes", nargs="+", type=int, default=[4, 8])
    args = parser.parse_args()
    print(json.dumps(plan(args.beam_sizes), indent=2))


if __name__ == "__main__":
    main()
