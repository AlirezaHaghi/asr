"""Retrospective reconstruction; results require rerunning.

Aggregate observed decoding rows; never substitute advertised model metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from jiwer import cer, process_words, wer


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def bootstrap_mean(values: list[float], repetitions: int, seed: int) -> list[float | None]:
    if not values:
        return [None, None]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = rng.choice(array, size=(repetitions, array.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return [round(float(low), 6), round(float(high), 6)]


def summarize(run_id: str, rows: list[dict], repetitions: int, seed: int) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    pairs = [(normalize(r["reference"]), normalize(r["hypothesis"])) for r in ok if r.get("reference") is not None]
    s = d = i = h = 0
    for ref, hyp in pairs:
        alignment = process_words(ref, hyp)
        s += alignment.substitutions; d += alignment.deletions; i += alignment.insertions; h += alignment.hits
    total_audio = sum(float(r.get("audio_duration_s") or 0) for r in ok)
    total_inference = sum(float(r.get("inference_s") or 0) for r in ok)
    utterance_wers = [float(r["utterance_wer"]) for r in ok if r.get("utterance_wer") is not None]
    return {
        "run_id": run_id,
        "comparison_family": rows[0].get("comparison_family") if rows else None,
        "model_id": rows[0].get("model_id") if rows else None,
        "generate_kwargs": rows[0].get("generate_kwargs") if rows else None,
        "rows": len(rows), "successful": len(ok), "failed": len(rows) - len(ok),
        "scored": len(pairs),
        "corpus_wer": round(wer([p[0] for p in pairs], [p[1] for p in pairs]), 6) if pairs else None,
        "corpus_cer": round(cer([p[0] for p in pairs], [p[1] for p in pairs]), 6) if pairs else None,
        "substitutions": s, "deletions": d, "insertions": i, "hits": h,
        "mean_utterance_wer": round(float(np.mean(utterance_wers)), 6) if utterance_wers else None,
        "mean_utterance_wer_bootstrap_95": bootstrap_mean(utterance_wers, repetitions, seed),
        "weighted_real_time_factor": round(total_inference / total_audio, 6) if total_audio else None,
        "audio_s": round(total_audio, 4), "inference_s": round(total_inference, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ablation_summary.json"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_jsonl(args.observations)
    latest = {(r["run_id"], r["id"]): r for r in raw}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in latest.values():
        groups[row["run_id"]].append(row)
    summaries = [summarize(run_id, groups[run_id], args.bootstrap, args.seed) for run_id in sorted(groups)]
    payload = {"experiment": "retrospective_decoding_ablation_summary", "source": str(args.observations.resolve()), "bootstrap_note": "95% percentile interval of mean per-utterance WER, not corpus WER", "runs": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
