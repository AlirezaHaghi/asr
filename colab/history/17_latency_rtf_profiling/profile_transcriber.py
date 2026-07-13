"""Retrospective reconstruction; results require rerunning.

Profile current-run ASR latency and RTF with explicit provenance. This is not
an original timestamped development artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
import traceback

import soundfile as sf
import torch
from transformers import pipeline


AUDIO_SUFFIXES = {".wav", ".flac", ".ogg"}


def fingerprint(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def synchronize(device: int) -> None:
    if device >= 0 and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", choices=("whisper", "ctc"), default="whisper")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--device", type=int, default=-1, help="-1 CPU, >=0 CUDA index")
    parser.add_argument("--pattern", default="**/*")
    parser.add_argument("--output", type=Path, default=Path("profile.json"))
    args = parser.parse_args()
    config = {"model": args.model, "model_type": args.model_type, "beam_size": args.beam_size, "device": args.device, "audio_dir": str(args.audio_dir.resolve()), "pattern": args.pattern}
    config_sha256 = fingerprint(config)
    payload = {"provenance": "retrospective reconstruction; rerun required", "config": config, "config_sha256": config_sha256, "load_seconds": None, "records": [], "fatal_error": None}
    try:
        if args.beam_size < 1:
            raise ValueError("beam-size must be >= 1")
        files = [path for path in sorted(args.audio_dir.glob(args.pattern)) if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES]
        if not files:
            raise FileNotFoundError(f"No supported audio files found under {args.audio_dir}")
        start_load = time.perf_counter()
        transcriber = pipeline("automatic-speech-recognition", model=args.model, device=args.device)
        synchronize(args.device)
        payload["load_seconds"] = time.perf_counter() - start_load
        for audio_path in files:
            record = {"audio": str(audio_path), "config_sha256": config_sha256, "model": args.model, "beam_size": args.beam_size, "device": args.device, "status": "pending"}
            try:
                info = sf.info(audio_path)
                duration = float(info.duration)
                synchronize(args.device)
                started = time.perf_counter()
                kwargs = {"generate_kwargs": {"num_beams": args.beam_size}} if args.model_type == "whisper" else {}
                result = transcriber(str(audio_path), **kwargs)
                synchronize(args.device)
                wall = time.perf_counter() - started
                record.update({"status": "ok", "duration_seconds": duration, "wall_seconds": wall, "rtf": wall / duration if duration > 0 else None, "text": result.get("text", "") if isinstance(result, dict) else str(result)})
            except Exception as exc:
                logging.exception("profiling failed for %s", audio_path)
                record.update({"status": "error", "exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)})
            payload["records"].append(record)
    except Exception as exc:
        logging.exception("profiling run failed")
        payload["fatal_error"] = {"exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=12)}
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.output}")
    return 1 if payload["fatal_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
