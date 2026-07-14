"""Retrospective reconstruction; rerun to validate unified ASR prediction JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {"run_id", "model_id", "family", "file_id", "reference", "hypothesis"}
OPTIONAL_DEFAULTS = {"duration_s": None, "inference_s": None, "error": None}


def normalize(row: dict) -> dict:
    output = {**OPTIONAL_DEFAULTS, **row}
    missing = sorted(REQUIRED - output.keys())
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if output["error"] is None and output["hypothesis"] is None:
        raise ValueError("successful row needs a hypothesis")
    output["reference"] = str(output["reference"] or "")
    output["hypothesis"] = str(output["hypothesis"] or "")
    return output


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "convert"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    valid, errors = [], []
    for index, row in enumerate(load(args.input), 1):
        try:
            valid.append(normalize(row))
        except Exception as exc:
            errors.append({"line": index, "error": str(exc)})
    report = {"input": str(args.input), "valid": len(valid), "invalid": len(errors), "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.action == "convert" and args.output:
        args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in valid), encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
