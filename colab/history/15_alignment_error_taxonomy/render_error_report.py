"""Retrospective reconstruction; results require rerunning.

Render a supplied taxonomy artifact as Markdown. This is not an original
timestamped development artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def escape(value: object) -> str:
    return str(value if value is not None else "∅").replace("|", "\\|").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("taxonomy", type=Path)
    parser.add_argument("--output", type=Path, default=Path("error_report.md"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    try:
        payload = json.loads(args.taxonomy.read_text(encoding="utf-8"))
        errors = payload.get("errors")
        if not isinstance(errors, list):
            raise ValueError("taxonomy JSON has no errors list")
        lines = ["# Generated error review", "", f"Source: `{escape(payload.get('source', 'unknown'))}`", "", "This report is computed from the selected taxonomy JSON; rerun the classifier for new predictions.", "", "## Counts", ""]
        for group, values in payload.get("counts", {}).items():
            lines.append(f"- {escape(group)}: {escape(dict(values))}")
        lines += ["", "## Aligned errors", "", "| record | op | reference | hypothesis | category | severity |", "|---|---|---|---|---|---|"]
        for error in errors[: max(0, args.limit)]:
            lines.append("| " + " | ".join(escape(error.get(key)) for key in ("record_id", "operation", "reference", "hypothesis", "category", "severity")) + " |")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
        return 0
    except Exception:
        logging.exception("report rendering failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
