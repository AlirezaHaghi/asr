"""Retrospective reconstruction; results require rerunning.

Cache supplied predictions atomically with current-config provenance. This is
not an original timestamped development artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import tempfile
import traceback


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_rows(payload: object) -> list:
    if isinstance(payload, list): return payload
    if isinstance(payload, dict):
        for key in ("records", "predictions", "items"):
            if isinstance(payload.get(key), list): return payload[key]
    raise ValueError("predictions must be a list or contain records/predictions/items")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, prefix=".tmp-", suffix=".json") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            temporary = Path(stream.name)
        temporary.replace(path)
    finally:
        if temporary and temporary.exists(): temporary.unlink()


def log_exception(path: Path, exc: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=12)}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--run-id")
    parser.add_argument("--error-log", type=Path, default=Path("cache_errors.jsonl"))
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
        if not isinstance(config, dict): raise ValueError("config must be a JSON object")
        rows = prediction_rows(predictions)
        run_id = args.run_id or config.get("run_id")
        if not run_id: raise ValueError("run_id is required in config or --run-id")
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id)).strip("._")
        if not safe_run_id: raise ValueError("run_id has no safe filename characters")
        config_sha = canonical_hash(config)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = args.cache_dir / f"predictions_{safe_run_id}_{config_sha[:12]}.json"
        atomic_json(cache_path, predictions)
        prediction_sha = file_hash(cache_path)
        generate = config.get("generate_kwargs") or config.get("generation") or {}
        entry = {"run_id": str(run_id), "created_at_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": config_sha, "predictions_sha256": prediction_sha, "predictions_file": cache_path.name, "records": len(rows), "model": config.get("model_id", config.get("model")), "beam_size": generate.get("num_beams", config.get("beam_size")), "config_source": str(args.config), "predictions_source": str(args.predictions)}
        manifest_path = args.cache_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"schema_version": 1, "entries": []}
        if not isinstance(manifest.get("entries"), list): raise ValueError("manifest entries must be a list")
        manifest["entries"] = [item for item in manifest["entries"] if item.get("config_sha256") != config_sha]
        manifest["entries"].append(entry)
        atomic_json(manifest_path, manifest)
        print(f"cached {len(rows)} records at {cache_path}")
        return 0
    except Exception as exc:
        logging.exception("prediction caching failed")
        try: log_exception(args.error_log, exc)
        except Exception: logging.exception("could not append cache error log")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
