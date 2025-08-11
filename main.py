import sys
from pathlib import Path
import os
from tqdm import tqdm
from asr_engine import transcribe

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <source_audio_path>")
        sys.exit(1)
    src_dir_name = sys.argv[1]

    src_path = Path(src_dir_name)
    if not src_path.is_dir():
        print(f"Error: {src_path} is not a valid directory.")
        sys.exit(1)

    dest_dir_name = src_dir_name + "-transcripts"
    dest_path = Path(dest_dir_name)
    dest_path.mkdir(exist_ok=True)

    audio_files_raw = [
        os.path.join(src_path, f)
        for f in os.listdir(src_path)
        if f.lower().endswith(("_clean.wav"))
    ]

    processed_files = set(
        os.path.splitext(os.path.basename(f))[0]
        for f in os.listdir(dest_path)
        if f.lower().endswith(".txt")
    )

    # Filter out already processed files and store (path, base_name) tuples
    files_to_process_info = []
    for file_path in audio_files_raw:
        file_name_base = os.path.splitext(os.path.basename(file_path))[0].replace("_clean", "")
        if file_name_base not in processed_files:
            files_to_process_info.append((file_path, file_name_base))

    # --- Crucial Change: Define Batch Size ---
    BATCH_SIZE = 8 # <<<<<<<<< Experiment with this value! (e.g., 4, 8, 16, 32, 64)
                   # Your RTX 5090 with ample VRAM should allow for large batches.

    # Transcribe files in batches with progress bar
    with tqdm(total=len(files_to_process_info), desc="Transcribing audio files") as pbar:
        for i in range(0, len(files_to_process_info), BATCH_SIZE):
            batch_info = files_to_process_info[i : i + BATCH_SIZE]
            batch_src_paths = [info[0] for info in batch_info] # List of file paths
            batch_file_names_base = [info[1] for info in batch_info] # List of base names

            if not batch_src_paths:
                continue

            # Transcribe the batch of audio files
            transcriptions = transcribe(batch_src_paths)

            # Save the results for each file in the batch
            for j, transcript_text in enumerate(transcriptions):
                file_name = batch_file_names_base[j]
                # print(f"Processed: {batch_src_paths[j]}") # Optional: for detailed output
                with open(dest_path / f"{file_name}.txt", "w", encoding="utf-8") as f:
                    f.write(transcript_text)
                pbar.update(1) # Update progress bar for each file transcribed