"""Retrospective reconstruction; rerun a restart-friendly FastConformer WAV batch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def files(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted(path for path in root.glob(pattern) if path.is_file())


def done_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    completed = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("status") == "ok":
                completed.add(row["file"])
        except json.JSONDecodeError:
            continue
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", default="niclaswue/youtube-atc-fastconformer")
    parser.add_argument("--output", type=Path, default=Path("fastconformer_run.jsonl"))
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    from nemo.collections.asr.models import ASRModel

    model = ASRModel.from_pretrained(args.model)
    completed = done_ids(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for path in files(args.root, args.recursive):
            key = str(path.resolve())
            if key in completed:
                continue
            started = time.perf_counter()
            try:
                hypothesis = model.transcribe([str(path)])[0]
                row = {"file": key, "status": "ok", "text": str(getattr(hypothesis, "text", hypothesis)), "seconds": round(time.perf_counter() - started, 3)}
            except Exception as exc:  # خراب شد؟ می‌نویسیم، رد نمی‌شیم انگار نه انگار
                row = {"file": key, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
