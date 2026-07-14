"""Retrospective reconstruction from the final notebook/report; results require a rerun.

Run bounded transcript and audio quality checks on ATCO2.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset

DATASET_ID = "Jzuluaga/atco2_corpus_1h"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=871)
    parser.add_argument("--output", type=Path, default=Path("quality.json"))
    args = parser.parse_args()
    ds = load_dataset(DATASET_ID, split=args.split)
    texts, durations, rates = [], [], Counter()
    missing_text = missing_audio = nonfinite_audio = 0
    unusual_examples = []
    for index, item in enumerate(ds):
        if index >= args.limit:
            break
        text = str(item.get("text", item.get("transcription", ""))).strip()
        if not text:
            missing_text += 1
        texts.append(text.lower())
        if re.search(r"[^a-zA-Z0-9\s.,'?!:/()-]", text) and len(unusual_examples) < 20:
            unusual_examples.append({"index": index, "text": text})
        audio = item.get("audio") or {}
        array = audio.get("array")
        rate = int(audio.get("sampling_rate") or 0)
        if array is None or rate <= 0:
            missing_audio += 1
            continue
        values = np.asarray(array)
        if not np.isfinite(values).all():
            nonfinite_audio += 1
        durations.append(len(values) / rate)
        rates[rate] += 1
    duplicate_rows = sum(count - 1 for count in Counter(texts).values() if text and count > 1)
    payload = {
        "dataset_id": DATASET_ID, "split": args.split, "rows_checked": len(texts),
        "missing_text": missing_text, "missing_audio": missing_audio,
        "nonfinite_audio": nonfinite_audio, "duplicate_normalized_text_rows": duplicate_rows,
        "sampling_rates": dict(rates),
        "duration_s": ({"min": min(durations), "mean": float(np.mean(durations)),
                         "median": float(np.median(durations)), "max": max(durations)} if durations else None),
        "unusual_character_examples": unusual_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("rows_checked", "missing_text", "missing_audio")}, indent=2))


if __name__ == "__main__":
    main()
