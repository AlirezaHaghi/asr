"""Retrospective reconstruction; results require rerunning.

Aggregate prediction metrics with matching current-run config provenance. This
is not an original timestamped development artifact.
"""

# خودمونی: corpus WER رو با میانگین WER جمله‌ها قاطی نکن.

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
from pathlib import Path


def canonical_fingerprint(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def payload_rows(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "predictions", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("Unsupported prediction schema")


def text(row: dict, choices: tuple[str, ...]) -> str:
    for key in choices:
        if row.get(key) is not None:
            return str(row[key]).lower().strip()
    raise KeyError(f"Missing text field from {choices}")


def distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(previous[j - 1] if left == right else 1 + min(previous[j - 1], previous[j], current[-1]))
        previous = current
    return previous[-1]


def determine_run_id(path: Path, payload: object) -> str:
    if isinstance(payload, dict) and payload.get("run_id"):
        return str(payload["run_id"])
    stem = path.stem
    return stem.removeprefix("predictions_")


def matching_config(config_dir: Path, run_id: str) -> tuple[Path, dict]:
    exact = config_dir / f"config_{run_id}.json"
    candidates = [exact] if exact.exists() else sorted(config_dir.glob("*.json"))
    for path in candidates:
        config = json.loads(path.read_text(encoding="utf-8"))
        if path == exact or str(config.get("run_id", "")) == run_id:
            return path, config
    raise FileNotFoundError(f"No config matched current run_id={run_id!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_glob")
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("run_table.json"))
    args = parser.parse_args()
    try:
        paths = [Path(name) for name in sorted(glob.glob(args.prediction_glob))]
        if not paths:
            raise FileNotFoundError(f"No prediction files matched {args.prediction_glob!r}")
        results = []
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows, run_id = payload_rows(payload), determine_run_id(path, payload)
            config_path, config = matching_config(args.config_dir, run_id)
            words = errors = perfect = 0
            for row in rows:
                ref = text(row, ("reference", "reference_raw", "ref", "text")).split()
                hyp = text(row, ("hypothesis", "hypothesis_raw", "hyp", "prediction")).split()
                words += len(ref)
                errors += distance(ref, hyp)
                perfect += ref == hyp
            generate = config.get("generate_kwargs") or config.get("generation") or {}
            results.append({"run_id": run_id, "predictions": str(path), "config": str(config_path), "config_sha256": canonical_fingerprint(config), "beam_size": generate.get("num_beams", config.get("beam_size")), "samples": len(rows), "perfect": perfect, "reference_words": words, "errors": errors, "wer": errors / words if words else None})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"runs": results}, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
        return 0
    except Exception:
        logging.exception("run aggregation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
