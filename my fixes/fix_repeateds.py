import os
import sys
from pathlib import Path
from tqdm import tqdm

from asr_engine import transcribe


def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_repeateds.py <number>")
        print("Example: python fix_repeateds.py 10")
        sys.exit(1)

    try:
        target_number = int(sys.argv[1])
    except ValueError:
        print("Error: Number must be an integer")
        sys.exit(1)

    with open("consecutive_repeated_words.txt", "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    filtered_lines = []
    for line in lines:
        full_path = line.split("\\")
        if len(full_path) >= 2:
            dir_name = full_path[0]
            if dir_name.startswith(f"clean{target_number}-"):
                filtered_lines.append(line)

    if not filtered_lines:
        print(f"No files found for clean{target_number}")
        return

    for line in tqdm(filtered_lines, desc=f"clean{target_number}"):
        full_path = line.split("\\")
        dir_name = full_path[0]
        file_name = full_path[1]
        src_dir_name = dir_name.split("-")[0]

        dest_dir_name = src_dir_name + "-transcripts"
        dest_path = Path(dest_dir_name)
        dest_path.mkdir(exist_ok=True)

        src_path = Path(src_dir_name).joinpath(file_name.replace(".txt", "_clean.wav"))
        if not src_path.exists():
            print(f"File {src_path} does not exist")
            continue

        result = transcribe(str(src_path))

        with open(dest_path / f"{file_name}.txt", "w", encoding="utf-8") as f:
            f.write(result)


if __name__ == "__main__":
    main()
