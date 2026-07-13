"""Retrospective reconstruction; rerun to check train/eval manifest overlap."""

# خودمونی: overlap رو دیدی، قایمش نکن؛ همون اول گزارش بده.

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def text_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def audio_key(row: dict) -> str | None:
    existing = row.get("audio_sha256") or row.get("sha256")
    if existing:
        return str(existing)
    path = row.get("audio_path") or row.get("path")
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index(records: list[dict], key_fn) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for number, row in enumerate(records):
        key = key_fn(row)
        if key:
            result[str(key)].append(str(row.get("id", number)))
    return result


def overlaps(left: dict[str, list[str]], right: dict[str, list[str]]) -> list[dict]:
    return [{"key": key, "train_ids": left[key], "eval_ids": right[key]} for key in sorted(left.keys() & right.keys())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("eval", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    train, evaluate = rows(args.train), rows(args.eval)
    checks = {
        "id": overlaps(index(train, lambda r: r.get("id")), index(evaluate, lambda r: r.get("id"))),
        "audio": overlaps(index(train, audio_key), index(evaluate, audio_key)),
        "normalized_text": overlaps(index(train, lambda r: text_key(str(r.get("text") or r.get("reference") or ""))), index(evaluate, lambda r: text_key(str(r.get("text") or r.get("reference") or "")))),
    }
    report = {"train_rows": len(train), "eval_rows": len(evaluate), "overlap_counts": {name: len(value) for name, value in checks.items()}, "overlaps": checks}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
