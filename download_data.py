import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "Jzuluaga/atco2_corpus_1h"
AUDIO_DIR  = "eval_data/audio"
TEXT_DIR   = "eval_data/ground_truth"


def setup_evaluation_dataset():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)

    dataset = load_dataset(DATASET_ID, split="test", trust_remote_code=True)

    for i, item in enumerate(tqdm(dataset, desc="saving")):
        audio_array  = item["audio"]["array"]
        sample_rate  = item["audio"]["sampling_rate"]
        text         = item.get("text", item.get("transcription", "")).strip()
        file_id      = item.get("id", f"sample_{i:04d}")

        sf.write(f"{AUDIO_DIR}/{file_id}.wav", audio_array, sample_rate)

        with open(f"{TEXT_DIR}/{file_id}.txt", "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    setup_evaluation_dataset()
