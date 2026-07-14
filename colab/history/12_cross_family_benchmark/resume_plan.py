"""Retrospective reconstruction; results require rerunning.

Build a checksum-verified cache resume plan for supplied configs. This is not
an original timestamped development artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
import logging
from pathlib import Path
import traceback


def config_hash(config: object) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_glob")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output", type=Path, default=Path("resume_plan.json"))
    args = parser.parse_args()
    plan = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "cache_dir": str(args.cache_dir), "runs": [], "manifest_error": None}
    try:
        config_paths = [Path(name) for name in sorted(glob.glob(args.config_glob))]
        if not config_paths: raise FileNotFoundError(f"No configs matched {args.config_glob!r}")
        manifest_path = args.cache_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"entries": []}
        entries = manifest.get("entries")
        if not isinstance(entries, list): raise ValueError("manifest entries must be a list")
        by_fingerprint = {str(entry.get("config_sha256")): entry for entry in entries if entry.get("config_sha256")}
        for config_path in config_paths:
            row = {"config": str(config_path), "action": "error"}
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if not isinstance(config, dict): raise ValueError("config is not an object")
                fingerprint = config_hash(config)
                generate = config.get("generate_kwargs") or config.get("generation") or {}
                row.update({"run_id": config.get("run_id", config_path.stem), "config_sha256": fingerprint, "model": config.get("model_id", config.get("model")), "beam_size": generate.get("num_beams", config.get("beam_size"))})
                entry = by_fingerprint.get(fingerprint)
                if not entry:
                    row.update({"action": "rerun", "reason": "no exact config fingerprint in manifest"})
                else:
                    cached = args.cache_dir / str(entry.get("predictions_file", ""))
                    if not cached.is_file():
                        row.update({"action": "rerun", "reason": "manifest cache file is missing"})
                    elif file_hash(cached) != entry.get("predictions_sha256"):
                        row.update({"action": "rerun", "reason": "prediction checksum mismatch"})
                    else:
                        row.update({"action": "reuse", "reason": "exact config fingerprint and prediction checksum match", "predictions": str(cached)})
            except Exception as exc:
                logging.exception("could not plan config %s", config_path)
                row["error"] = {"exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
            plan["runs"].append(row)
    except Exception as exc:
        logging.exception("resume planning failed")
        plan["manifest_error"] = {"exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=12)}
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 1 if plan["manifest_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
