"""Retrospective reconstruction; results require rerunning.

Compare supplied profile artifacts without mixing run provenance. This is not
an original timestamped development artifact.
"""

# خودمونی: load time و inference time رو توی یه عدد نریز.

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
import statistics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_glob")
    parser.add_argument("--output", type=Path, default=Path("profile_comparison.json"))
    args = parser.parse_args()
    try:
        paths = [Path(name) for name in sorted(glob.glob(args.profile_glob))]
        if not paths: raise FileNotFoundError(f"No files matched {args.profile_glob!r}")
        runs = []
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            config, fingerprint = payload.get("config"), payload.get("config_sha256")
            if not isinstance(config, dict) or not fingerprint: raise ValueError(f"{path} lacks config provenance")
            ok = [row for row in payload.get("records", []) if row.get("status") == "ok" and row.get("rtf") is not None]
            runs.append({"profile": str(path), "config_sha256": fingerprint, "model": config.get("model"), "beam_size": config.get("beam_size"), "device": config.get("device"), "successful_files": len(ok), "mean_rtf": statistics.fmean(float(row["rtf"]) for row in ok) if ok else None, "median_rtf": statistics.median(float(row["rtf"]) for row in ok) if ok else None})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"runs": runs, "note": "beam/model/device are taken from each profile's own config"}, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
        return 0
    except Exception:
        logging.exception("profile comparison failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
