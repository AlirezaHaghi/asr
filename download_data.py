"""
download_data.py — دانلود ATCO2-test-set-1h برای تست آفلاین
این فایل اختیاری است — evaluate.py مستقیم از HuggingFace بارگذاری می‌کند
"""
import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# ← همان dataset که evaluate.py استفاده می‌کند
DATASET_ID = "Jzuluaga/atco2_corpus_1h"
AUDIO_DIR  = "eval_data/audio"
TEXT_DIR   = "eval_data/ground_truth"


def setup_evaluation_dataset():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)

    print(f"دانلود {DATASET_ID} از HuggingFace...")
    dataset = load_dataset(DATASET_ID, split="test", trust_remote_code=True)
    print(f"{len(dataset)} نمونه پیدا شد")

    for i, item in enumerate(tqdm(dataset, desc="ذخیره")):
        audio_array  = item["audio"]["array"]
        sample_rate  = item["audio"]["sampling_rate"]
        text         = item.get("text", item.get("transcription", "")).strip()
        file_id      = item.get("id", f"sample_{i:04d}")

        sf.write(f"{AUDIO_DIR}/{file_id}.wav", audio_array, sample_rate)

        with open(f"{TEXT_DIR}/{file_id}.txt", "w", encoding="utf-8") as f:
            f.write(text)

    print(f"\nتمام! فایل‌ها در eval_data/")


if __name__ == "__main__":
    setup_evaluation_dataset()
