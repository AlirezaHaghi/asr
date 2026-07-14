"""Retrospective reconstruction; results require rerunning.

Execute a decoding grid on a fixed local audio manifest with failure logging.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from jiwer import cer, wer
from transformers import pipeline


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_configs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    configs = payload if isinstance(payload, list) else payload.get("configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("config file must contain a nonempty 'configs' list")
    return configs


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {(r["run_id"], r["id"]) for r in load_jsonl(path) if r.get("status") == "ok"}


def append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, default=Path("decoding_grid.json"))
    parser.add_argument("--output", type=Path, default=Path("observations.jsonl"))
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = [r for r in load_jsonl(args.manifest) if r.get("valid", True)]
    if args.max_items is not None:
        items = items[: args.max_items]
    configs = load_configs(args.config)
    model_ids = {c["model_id"] for c in configs}
    if len(model_ids) != 1:
        raise ValueError("this ablation isolates decoding; all configs must use one model_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.no_resume and args.output.exists():
        raise FileExistsError(f"refusing to append with --no-resume: {args.output}")
    done = set() if args.no_resume else completed_keys(args.output)
    model_id = next(iter(model_ids))
    recognizer = pipeline(
        "automatic-speech-recognition", model=model_id,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        chunk_length_s=30,
    )
    for config in configs:
        for index, item in enumerate(items, start=1):
            key = (config["run_id"], item["id"])
            if key in done:
                continue
            started = time.perf_counter()
            base = {
                "run_id": config["run_id"], "comparison_family": config.get("comparison_family"),
                "id": item["id"], "audio_path": item["audio_path"], "audio_sha256": item.get("sha256"),
                "audio_duration_s": item.get("duration_s"), "reference": item.get("reference"),
                "model_id": model_id, "generate_kwargs": config["generate_kwargs"],
            }
            try:
                result = recognizer(item["audio_path"], generate_kwargs=config["generate_kwargs"])
                elapsed = time.perf_counter() - started
                hypothesis = result.get("text", "").strip()
                record = {**base, "status": "ok", "hypothesis": hypothesis, "inference_s": round(elapsed, 4), "real_time_factor": round(elapsed / item["duration_s"], 4) if item.get("duration_s") else None, "error": None}
                if item.get("reference") is not None:
                    ref, hyp = normalize(item["reference"]), normalize(hypothesis)
                    record.update({"utterance_wer": round(wer(ref, hyp), 6), "utterance_cer": round(cer(ref, hyp), 6)})
            except Exception as exc:
                record = {**base, "status": "error", "hypothesis": None, "inference_s": round(time.perf_counter() - started, 4), "real_time_factor": None, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8)}
            append(args.output, record)
            print(f"[{config['run_id']}] {index}/{len(items)} {record['status']} {item['id']}")


if __name__ == "__main__":
    main()
