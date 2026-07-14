"""Retrospective reconstruction; results require rerunning.

Summarize only records observed in a recreated directory batch run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from jiwer import cer, process_words, wer


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def scoring(records: list[dict]) -> dict | None:
    pairs = [(normalize(r["reference"]), normalize(r["hypothesis"])) for r in records if r.get("reference") is not None and r.get("hypothesis") is not None]
    if not pairs:
        return None
    refs, hyps = zip(*pairs)
    substitutions = deletions = insertions = hits = 0
    for ref, hyp in pairs:
        observed = process_words(ref, hyp)
        substitutions += observed.substitutions
        deletions += observed.deletions
        insertions += observed.insertions
        hits += observed.hits
    return {
        "scored_items": len(pairs),
        "wer": round(wer(list(refs), list(hyps)), 6),
        "cer": round(cer(list(refs), list(hyps)), 6),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "hits": hits,
        "normalization": "lowercase and collapse whitespace only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_records = load_jsonl(args.predictions)
    latest_by_id = {r["id"]: r for r in all_records}
    records = list(latest_by_id.values())
    successful = [r for r in records if r.get("status") == "ok"]
    durations = [float(r["audio_duration_s"]) for r in successful if r.get("audio_duration_s") is not None]
    inference = [float(r["inference_s"]) for r in successful if r.get("inference_s") is not None]
    total_audio, total_inference = sum(durations), sum(inference)
    summary = {
        "experiment": "retrospective_directory_batch_summary",
        "source": str(args.predictions.resolve()),
        "raw_lines": len(all_records),
        "latest_unique_ids": len(records),
        "status": dict(Counter(r.get("status", "missing") for r in records)),
        "models": dict(Counter(r.get("model_id", "missing") for r in records)),
        "audio_hours": round(total_audio / 3600, 6),
        "inference_hours": round(total_inference / 3600, 6),
        "weighted_real_time_factor": round(total_inference / total_audio, 6) if total_audio else None,
        "median_item_inference_s": round(statistics.median(inference), 6) if inference else None,
        "scoring": scoring(successful),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
