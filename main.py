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




    audio_files = [
        os.path.join(src_path, f)
        for f in os.listdir(src_path)
        if f.lower().endswith(("_clean.wav"))
    ]

    processed_files = set(
        os.path.splitext(os.path.basename(f))[0]
        for f in os.listdir(dest_path)
        if f.lower().endswith(".txt")
    )

    # Transcribe files with progress bar
    with tqdm(total=len(audio_files), desc="Transcribing audio files") as pbar:
        for file in audio_files:
            file_name = os.path.splitext(os.path.basename(file))[0].replace("_clean", "")
            if file_name in processed_files:
                pbar.update(1)
                continue  # Skip already processed files
            
            pbar.update(1)
            success = transcribe(file)
            print(file)
            with open(dest_path / f"{file_name}.txt", "w", encoding="utf-8") as f:
                f.write(success)