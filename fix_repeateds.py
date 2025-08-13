# sample lien:
# clean10-transcripts\VIPER31_1398-3-26_17-32-38_124.500000MHz_AM.txt
import os
import sys
from pathlib import Path
from tqdm import tqdm

from asr_engine import transcribe


def main():
    # Get command line argument
    if len(sys.argv) != 2:
        print("Usage: python fix_repeateds.py <number>")
        print("Example: python fix_repeateds.py 10")
        sys.exit(1)

    try:
        target_number = int(sys.argv[1])
    except ValueError:
        print("Error: Number must be an integer")
        sys.exit(1)

    print(f"Processing files from clean{target_number} directory...")

    # Read all lines first to count total files
    with open("consecutive_repeated_words.txt", "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Filter lines that match the target number
    filtered_lines = []
    for line in lines:
        full_path = line.split("\\")
        if len(full_path) >= 2:
            dir_name = full_path[0]
            # Extract number from directory name (e.g., "clean10-transcripts" -> "10")
            if dir_name.startswith(f"clean{target_number}-"):
                filtered_lines.append(line)

    print(f"Found {len(filtered_lines)} files matching clean{target_number}")

    if not filtered_lines:
        print(f"No files found for clean{target_number}")
        return

    # Process filtered files with progress bar
    for line in tqdm(filtered_lines, desc=f"Processing clean{target_number} files"):
        full_path = line.split("\\")
        # in production there would be some files like clean7 clean10 and so on.
        # we should extract the dir name
        dir_name = full_path[0]
        file_name = full_path[1]
        src_dir_name = dir_name.split("-")[0]

        dest_dir_name = src_dir_name + "-transcripts"
        dest_path = Path(dest_dir_name)
        dest_path.mkdir(exist_ok=True)

        # then we must generate the file path
        src_path = Path(src_dir_name).joinpath(file_name.replace(".txt", "_clean.wav"))
        # check if the file exists
        if not src_path.exists():
            print(f"File {src_path} does not exist")
            continue

        # if it found then we should pass it to the transcribe function
        # print(src_path)
        result = transcribe(str(src_path))
        # save the result to the file with relevant directory like the main.py

        with open(dest_path / f"{file_name}.txt", "w", encoding="utf-8") as f:
            f.write(result)


if __name__ == "__main__":
    main()
