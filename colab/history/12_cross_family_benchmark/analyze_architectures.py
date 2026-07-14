"""Retrospective reconstruction; results require a rerun.

Analyze observed Whisper/CTC alignments, timing, and corpus-level errors.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from jiwer import cer, process_words, wer


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def alignment_patterns(ref: str, hyp: str, counter: Counter, examples: list[dict], item_id: str) -> tuple[int, int, int, int]:
    aligned = process_words(ref, hyp)
    s, d, i, h = aligned.substitutions, aligned.deletions, aligned.insertions, aligned.hits
    ref_words, hyp_words = ref.split(), hyp.split()
    for chunk in aligned.alignments[0]:
        if chunk.type == "equal":
            continue
        ref_span = " ".join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx]) or "[empty]"
        hyp_span = " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]) or "[empty]"
        counter[(chunk.type, ref_span, hyp_span)] += 1
        if len(examples) < 50:
            examples.append({"id": item_id, "type": chunk.type, "reference_span": ref_span, "hypothesis_span": hyp_span, "reference": ref, "hypothesis": hyp})
    return s, d, i, h


def summarize(run_id: str, rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    pairs = [(r["reference_normalized"], r["hypothesis_normalized"], r["id"]) for r in ok if r.get("reference_normalized") is not None]
    totals = np.zeros(4, dtype=np.int64)
    patterns: Counter = Counter()
    examples: list[dict] = []
    for ref, hyp, item_id in pairs:
        totals += alignment_patterns(ref, hyp, patterns, examples, item_id)
    audio_s = sum(float(r.get("audio_duration_s") or 0) for r in ok)
    inference_s = sum(float(r.get("inference_s") or 0) for r in ok)
    top = [
        {"type": key[0], "reference_span": key[1], "hypothesis_span": key[2], "count": count}
        for key, count in patterns.most_common(25)
    ]
    refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "run_id": run_id, "architecture": rows[0].get("architecture") if rows else None, "model_id": rows[0].get("model_id") if rows else None,
        "rows": len(rows), "successful": len(ok), "failed": len(rows) - len(ok), "scored": len(pairs),
        "corpus_wer": round(wer(refs, hyps), 6) if pairs else None,
        "corpus_cer": round(cer(refs, hyps), 6) if pairs else None,
        "substitutions": int(totals[0]), "deletions": int(totals[1]), "insertions": int(totals[2]), "hits": int(totals[3]),
        "perfect_transcriptions": sum(ref == hyp for ref, hyp, _ in pairs),
        "weighted_real_time_factor": round(inference_s / audio_s, 6) if audio_s else None,
        "top_alignment_patterns": top,
        "first_error_examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", type=Path, default=Path("architecture_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_jsonl(args.observations)
    latest = {(r["run_id"], r["id"]): r for r in raw}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in latest.values(): groups[row["run_id"]].append(row)
    runs = [summarize(run_id, groups[run_id]) for run_id in sorted(groups)]
    payload = {"experiment": "retrospective_whisper_ctc_summary", "source": str(args.observations.resolve()), "normalization": "common stored normalized fields", "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
