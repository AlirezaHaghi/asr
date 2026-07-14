"""Retrospective reconstruction; results require rerunning.

Fingerprint a supplied JSON configuration canonically. This is not an original
timestamped development artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path


def canonical_bytes(config: object) -> bytes:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("config_fingerprint.json"))
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict): raise ValueError("config must be a JSON object")
        payload = {"source": str(args.config), "config_sha256": hashlib.sha256(canonical_bytes(config)).hexdigest(), "fingerprinted_at_utc": datetime.now(timezone.utc).isoformat(), "canonicalization": "UTF-8 JSON; keys sorted; compact separators; NaN forbidden"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
        return 0
    except Exception:
        logging.exception("config fingerprinting failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
