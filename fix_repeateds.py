# sample lien:
# clean10-transcripts\VIPER31_1398-3-26_17-32-38_124.500000MHz_AM.txt
import os
from pathlib import Path

from asr_engine import transcribe

with open("consecutive_repeated_words.txt", "r") as f:
    for line in f:
        line = line.strip()
        full_path = line.split("\\")
        print(full_path)
        # in produciton there would be some files like clean10 clean7 and so on.
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
            break

        # if it fount then we should pass it to the transcribe function
        result = transcribe(str(src_path))
        # save the result to the file with relevant directory lilke the main.py

        with open(dest_path / f"{file_name}.txt", "w", encoding="utf-8") as f:
            f.write(result)
