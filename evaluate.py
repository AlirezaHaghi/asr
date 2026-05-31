"""
evaluate.py — Benchmark روی ATCO2-test-set-1h
------------------------------------------------
اجرا:
  python evaluate.py

خروجی‌ها (در پوشه eval_output/):
  benchmark_results.json   ← WER/CER + S/D/I دو مدل
  predictions_run1.json    ← خروجی مدل ۱
  predictions_run2.json    ← خروجی مدل ۲
  error_analysis.txt       ← ۲۰+ نمونه دسته‌بندی‌شده
"""

import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from jiwer import wer, cer, process_words

from asr_engine import get_pipe
from text_normalizer import normalize_for_wer

# ── تنظیمات ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("eval_output")
OUTPUT_DIR.mkdir(exist_ok=True)

RUNS = [
    {
        "name": "jacktol-whisper-medium.en-ATC (beam=5)",
        "model_id": "jacktol/whisper-medium.en-fine-tuned-for-ATC",
        "beam_size": 5,
    },
    {
        # مقایسه با beam_size=1 (greedy) روی همان مدل — سریع‌تر، برای نشان دادن تفاوت
        "name": "jacktol-whisper-medium.en-ATC (beam=1, greedy)",
        "model_id": "jacktol/whisper-medium.en-fine-tuned-for-ATC",
        "beam_size": 1,
    },
]

# ── بارگذاری دیتاست ───────────────────────────────────────────────────────────
def load_atco2():
    print("بارگذاری ATCO2-test-set-1h از HuggingFace...")
    ds = load_dataset("Jzuluaga/atco2_corpus_1h", split="test", trust_remote_code=True)
    print(f"  {len(ds)} نمونه بارگذاری شد")
    return ds


# ── inference روی یک run ─────────────────────────────────────────────────────
def run_inference(dataset, run_cfg: dict) -> list[dict]:
    model_id = run_cfg["model_id"]
    beam_size = run_cfg["beam_size"]
    name = run_cfg["name"]

    print(f"\n{'='*55}")
    print(f"مدل: {name}")
    print(f"دستگاه: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*55}")

    pipe = get_pipe(model_id)
    records = []
    t_total = time.time()

    for i, item in enumerate(dataset):
        audio = item["audio"]
        ref_raw = item.get("text", item.get("transcription", "")).strip()
        ref = normalize_for_wer(ref_raw)

        t0 = time.time()
        result = pipe(
            {"array": audio["array"], "sampling_rate": audio["sampling_rate"]},
            generate_kwargs={
                "language": "english",
                "task": "transcribe",
                "temperature": 0.0,
                "num_beams": beam_size,
                "condition_on_prev_text": False,
            },
        )
        elapsed = time.time() - t0

        hyp_raw = result["text"].strip()
        hyp = normalize_for_wer(hyp_raw)

        w = wer(ref, hyp) if ref else 0.0
        c = cer(ref, hyp) if ref else 0.0

        records.append({
            "id": item.get("id", f"sample_{i:04d}"),
            "reference": ref,
            "hypothesis": hyp,
            "hypothesis_raw": hyp_raw,
            "wer": round(w, 4),
            "cer": round(c, 4),
            "duration_s": round(len(audio["array"]) / audio["sampling_rate"], 2),
            "inference_s": round(elapsed, 2),
        })

        if (i + 1) % 50 == 0:
            so_far = np.mean([r["wer"] for r in records])
            print(f"  [{i+1}/{len(dataset)}] WER تاکنون: {so_far:.2%}")

    print(f"  زمان کل: {time.time()-t_total:.0f}s")
    return records


# ── محاسبه S/D/I ─────────────────────────────────────────────────────────────
def compute_sdi(records: list[dict]) -> dict:
    S = D = I = H = N = 0
    for r in records:
        out = process_words(r["reference"], r["hypothesis"])
        S += out.substitutions
        D += out.deletions
        I += out.insertions
        H += out.hits
        N += len(r["reference"].split())
    n = max(N, 1)
    return {
        "WER":  round((S + D + I) / n, 4),
        "CER":  round(cer([r["reference"] for r in records],
                          [r["hypothesis"] for r in records]), 4),
        "S": S, "D": D, "I": I, "H": H, "N": N,
        "S_rate": round(S / n, 4),
        "D_rate": round(D / n, 4),
        "I_rate": round(I / n, 4),
    }


# ── تحلیل خطا ────────────────────────────────────────────────────────────────
CALLSIGN_AIRLINES = {
    "lufthansa","swiss","iberia","ryanair","easyjet","delta","united",
    "american","british","air france","alitalia","klm","turkish",
}

def categorize(word: str) -> str:
    w = word.lower()
    if any(a in w for a in CALLSIGN_AIRLINES): return "callsign"
    if w in {"zero","one","two","three","four","five","six","seven","eight","nine",
             "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
             "seventeen","eighteen","nineteen","twenty","thirty","forty","fifty",
             "sixty","seventy","eighty","ninety","hundred"}: return "number"
    if w in {"flight","level"}: return "flight_level"
    if w in {"runway","approach","ils","localizer"}: return "runway"
    if w in {"descend","climb","maintain","heading","turn","contact","cleared",
             "squawk","frequency","report","expedite","hold"}: return "command"
    return "other"


def extract_errors(records: list[dict], n: int = 30) -> list[dict]:
    errors = []
    # بدترین نمونه‌ها اول
    for rec in sorted(records, key=lambda r: r["wer"], reverse=True):
        if len(errors) >= n: break
        ref_words = rec["reference"].split()
        hyp_words = rec["hypothesis"].split()
        try:
            out = process_words(rec["reference"], rec["hypothesis"])
        except Exception:
            continue
        for al in out.alignments[0]:
            if len(errors) >= n: break
            if al.type == "substitute":
                rw = ref_words[al.ref_start_idx] if al.ref_start_idx < len(ref_words) else ""
                hw = hyp_words[al.hyp_start_idx] if al.hyp_start_idx < len(hyp_words) else ""
                errors.append({
                    "id": rec["id"], "type": "S",
                    "category": categorize(rw),
                    "ref_word": rw, "hyp_word": hw,
                    "ref_sentence": rec["reference"],
                    "hyp_sentence": rec["hypothesis"],
                })
            elif al.type == "delete":
                rw = ref_words[al.ref_start_idx] if al.ref_start_idx < len(ref_words) else ""
                errors.append({
                    "id": rec["id"], "type": "D",
                    "category": categorize(rw),
                    "ref_word": rw, "hyp_word": "[حذف]",
                    "ref_sentence": rec["reference"],
                    "hyp_sentence": rec["hypothesis"],
                })
    return errors


def write_error_report(errors: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ATC-ASR Error Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"تعداد کل نمونه: {len(errors)}\n\n")

        cats = Counter(e["category"] for e in errors)
        f.write("توزیع دسته‌بندی:\n")
        for cat, n in cats.most_common():
            f.write(f"  {cat:<15}: {n}\n")

        f.write("\n" + "=" * 60 + "\n\n")
        for e in errors:
            f.write(f"[{e['type']}] دسته: {e['category']}\n")
            f.write(f"  REF: {e['ref_sentence']}\n")
            f.write(f"  HYP: {e['hyp_sentence']}\n")
            f.write(f"  خطا: '{e['ref_word']}' → '{e['hyp_word']}'\n\n")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    dataset = load_atco2()
    summary = []

    for i, run in enumerate(RUNS, 1):
        records = run_inference(dataset, run)

        # ذخیره predictions
        pred_path = OUTPUT_DIR / f"predictions_run{i}.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        sdi = compute_sdi(records)
        wer_list = [r["wer"] for r in records]

        result = {
            "run_name": run["name"],
            "model_id": run["model_id"],
            "beam_size": run["beam_size"],
            "samples": len(records),
            **sdi,
            "wer_mean": round(np.mean(wer_list), 4),
            "wer_median": round(float(np.median(wer_list)), 4),
            "wer_std": round(float(np.std(wer_list)), 4),
            "perfect_transcriptions": sum(1 for w in wer_list if w == 0.0),
        }
        summary.append(result)

        print(f"\nنتایج {run['name']}:")
        print(f"  WER: {sdi['WER']:.2%} | CER: {sdi['CER']:.2%}")
        print(f"  S:{sdi['S']} D:{sdi['D']} I:{sdi['I']} (از {sdi['N']} کلمه)")

    # ذخیره خلاصه benchmark
    with open(OUTPUT_DIR / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # جدول نهایی
    print(f"\n{'='*65}")
    print(f"{'مدل':<45} {'WER':>6} {'CER':>6} {'S':>4} {'D':>4} {'I':>4}")
    print("-" * 65)
    for s in summary:
        print(f"{s['run_name']:<45} {s['WER']:>6.2%} {s['CER']:>6.2%} "
              f"{s['S']:>4} {s['D']:>4} {s['I']:>4}")

    # Error Analysis از run اول
    with open(OUTPUT_DIR / "predictions_run1.json", encoding="utf-8") as f:
        records1 = json.load(f)
    errors = extract_errors(records1, n=30)
    write_error_report(errors, OUTPUT_DIR / "error_analysis.txt")

    print(f"\n✓ خروجی‌ها در: {OUTPUT_DIR}/")
    print(f"  benchmark_results.json")
    print(f"  predictions_run1.json & run2.json")
    print(f"  error_analysis.txt ({len(errors)} نمونه)")


if __name__ == "__main__":
    main()
