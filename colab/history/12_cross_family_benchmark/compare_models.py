"""Retrospective reconstruction; results require a rerun.

Compare one ATC Whisper model and one ATC XLS-R CTC model on a fixed manifest.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
import traceback
from pathlib import Path

import torch
from jiwer import cer, wer
from transformers import pipeline

WHISPER_MODEL = "jacktol/whisper-large-v3-finetuned-for-ATC"
CTC_MODEL = "Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim"
DIGITS = {str(i): word for i, word in enumerate("zero one two three four five six seven eight nine".split())}
REPLACEMENTS = [
    (r"\bniner\b", "nine"), (r"\btree\b", "three"),
    (r"\bfife\b", "five"), (r"\balfa\b", "alpha"),
    (r"\bryan\s+air\b", "ryanair"), (r"\bspeed\s+bird\b", "speedbird"),
    (r"\beuro\s+wings\b", "eurowings"),
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    value = text.lower().strip()
    digit_words = lambda raw: " ".join(DIGITS[digit] for digit in raw)
    value = re.sub(r"\bfl\s*[- ]?(\d{2,3})\b", lambda m: "flight level " + digit_words(m.group(1)), value)
    value = re.sub(
        r"\brwy\s*[- ]?(\d{1,2})([lrc]?)\b",
        lambda m: "runway " + digit_words(m.group(1)) + ({"l": " left", "r": " right", "c": " center"}.get(m.group(2), "")),
        value,
    )
    value = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: digit_words(m.group(1)) + " decimal " + digit_words(m.group(2)), value)
    value = re.sub(r"\b\d+\b", lambda m: digit_words(m.group(0)), value)
    for pattern, replacement in REPLACEMENTS:
        value = re.sub(pattern, replacement, value)
    value = re.sub(r"[^a-z' ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def completed(path: Path) -> set[tuple[str, str]]:
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
    parser.add_argument("--whisper-model", default=WHISPER_MODEL)
    parser.add_argument("--ctc-model", default=CTC_MODEL)
    parser.add_argument("--whisper-beams", type=int, default=5)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--output", type=Path, default=Path("architecture_observations.jsonl"))
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = [r for r in load_jsonl(args.manifest) if r.get("valid", True)]
    if args.max_items is not None:
        items = items[: args.max_items]
    runs = [
        {"run_id": "whisper_encoder_decoder", "architecture": "Whisper encoder-decoder", "model_id": args.whisper_model, "generate_kwargs": {"language": "english", "task": "transcribe", "num_beams": args.whisper_beams, "do_sample": False}},
        {"run_id": "xlsr_ctc", "architecture": "XLS-R encoder with CTC head", "model_id": args.ctc_model, "generate_kwargs": {}},
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.no_resume and args.output.exists():
        raise FileExistsError(f"refusing to append with --no-resume: {args.output}")
    done = set() if args.no_resume else completed(args.output)
    for run in runs:
        recognizer = pipeline(
            "automatic-speech-recognition", model=run["model_id"],
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            chunk_length_s=30,
        )
        for index, item in enumerate(items, start=1):
            if (run["run_id"], item["id"]) in done:
                continue
            started = time.perf_counter()
            base = {
                "run_id": run["run_id"], "architecture": run["architecture"], "model_id": run["model_id"],
                "generate_kwargs": run["generate_kwargs"], "id": item["id"], "audio_path": item["audio_path"],
                "audio_sha256": item.get("sha256"), "audio_duration_s": item.get("duration_s"), "reference_raw": item.get("reference"),
            }
            try:
                kwargs = {"generate_kwargs": run["generate_kwargs"]} if run["generate_kwargs"] else {}
                output = recognizer(item["audio_path"], **kwargs)
                elapsed = time.perf_counter() - started
                hypothesis = output.get("text", "").strip()
                ref_norm = normalize(item["reference"]) if item.get("reference") is not None else None
                hyp_norm = normalize(hypothesis)
                record = {**base, "status": "ok", "hypothesis_raw": hypothesis, "reference_normalized": ref_norm, "hypothesis_normalized": hyp_norm, "inference_s": round(elapsed, 4), "real_time_factor": round(elapsed / item["duration_s"], 4) if item.get("duration_s") else None, "error": None}
                if ref_norm is not None:
                    record.update({"utterance_wer": round(wer(ref_norm, hyp_norm), 6), "utterance_cer": round(cer(ref_norm, hyp_norm), 6)})
            except Exception as exc:
                record = {**base, "status": "error", "hypothesis_raw": None, "reference_normalized": normalize(item["reference"]) if item.get("reference") is not None else None, "hypothesis_normalized": None, "inference_s": round(time.perf_counter() - started, 4), "real_time_factor": None, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8)}
            append(args.output, record)
            print(f"[{run['run_id']}] {index}/{len(items)} {record['status']} {item['id']}")
        del recognizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
