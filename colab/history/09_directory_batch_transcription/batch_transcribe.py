"""Retrospective reconstruction; results require rerunning.

Run restartable ATC Whisper inference over a local JSONL WAV manifest.
"""

# خودمونی: فایل خراب شد، خطاش رو بنویس و برو سراغ بعدی.

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from transformers import pipeline

DEFAULT_MODEL = "jacktol/whisper-large-v3-finetuned-for-ATC"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def successful_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["id"] for r in load_jsonl(path) if r.get("status") == "ok" and r.get("id")}


def append_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("predictions.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = [r for r in load_jsonl(args.manifest) if r.get("valid", True)]
    if args.max_items is not None:
        manifest = manifest[: args.max_items]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.no_resume else successful_ids(args.output)
    if args.no_resume and args.output.exists():
        raise FileExistsError(f"refusing to append with --no-resume: remove or rename {args.output}")
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        chunk_length_s=30,
    )
    generate_kwargs = {"language": "english", "task": "transcribe", "temperature": args.temperature, "num_beams": args.beams}
    for index, item in enumerate(manifest, start=1):
        if item["id"] in completed:
            print(f"[{index}/{len(manifest)}] skip successful {item['id']}")
            continue
        started = time.perf_counter()
        base = {
            "id": item["id"],
            "audio_path": item["audio_path"],
            "audio_sha256": item.get("sha256"),
            "audio_duration_s": item.get("duration_s"),
            "reference": item.get("reference"),
            "model_id": args.model,
            "generate_kwargs": generate_kwargs,
        }
        try:
            output = recognizer(item["audio_path"], generate_kwargs=generate_kwargs, return_timestamps=True)
            elapsed = time.perf_counter() - started
            record = {
                **base,
                "status": "ok",
                "hypothesis": output.get("text", "").strip(),
                "timestamps": output.get("chunks", []),
                "inference_s": round(elapsed, 4),
                "real_time_factor": round(elapsed / item["duration_s"], 4) if item.get("duration_s") else None,
                "error": None,
            }
        except Exception as exc:
            record = {
                **base,
                "status": "error",
                "hypothesis": None,
                "timestamps": [],
                "inference_s": round(time.perf_counter() - started, 4),
                "real_time_factor": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
        append_record(args.output, record)
        print(f"[{index}/{len(manifest)}] {record['status']} {item['id']}")


if __name__ == "__main__":
    main()
