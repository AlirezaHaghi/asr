import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

from asr_engine import transcribe
from text_normalizer import normalize_for_display


def process_pipeline(src_dir: str):
    src_path = Path(src_dir)
    if not src_path.is_dir():
        print(f"Error: {src_path} is not a valid directory.")
        sys.exit(1)

    dest_path = Path(src_dir + "-transcripts")
    dest_path.mkdir(exist_ok=True)
    logs_path = dest_path / "execution_logs.json"

    audio_files = [
        os.path.join(src_path, f)
        for f in os.listdir(src_path)
        if f.lower().endswith((".wav", ".mp3", ".flac"))
    ]

    processed = {
        os.path.splitext(f)[0]
        for f in os.listdir(dest_path)
        if f.lower().endswith(".txt")
    }

    execution_logs = []

    with tqdm(total=len(audio_files), desc="Transcribing") as pbar:
        for file_path in audio_files:
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            if file_name in processed:
                pbar.update(1)
                continue

            try:
                raw_text = transcribe(file_path)
                final_text = normalize_for_display(raw_text)

                with open(dest_path / f"{file_name}.txt", "w", encoding="utf-8") as f:
                    f.write(final_text)

                execution_logs.append({
                    "file": file_name,
                    "status": "success",
                    "raw": raw_text,
                    "normalized": final_text,
                })

            except Exception as e:
                execution_logs.append({
                    "file": file_name,
                    "status": "failed",
                    "error": str(e),
                })

            pbar.update(1)

    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(execution_logs, f, ensure_ascii=False, indent=2)

    ok = sum(1 for l in execution_logs if l["status"] == "success")
    fail = sum(1 for l in execution_logs if l["status"] == "failed")
    print(f"done: {ok} ok, {fail} failed")
    print(f"output: {dest_path}/")
    print(f"log: {logs_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <audio_folder>")
        sys.exit(1)
    process_pipeline(sys.argv[1])
