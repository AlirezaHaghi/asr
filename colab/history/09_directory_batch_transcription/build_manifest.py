"""Retrospective reconstruction; results require rerunning.

Build a deterministic, hash-bearing JSONL inventory for recursive WAV research.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_reference(relative_wav: Path, transcript_root: Path | None) -> tuple[str | None, str | None]:
    if transcript_root is None:
        return None, None
    transcript_path = (transcript_root / relative_wav).with_suffix(".txt")
    if not transcript_path.exists():
        return str(transcript_path.resolve()), None
    return str(transcript_path.resolve()), transcript_path.read_text(encoding="utf-8").strip()


def build_record(path: Path, root: Path, transcript_root: Path | None) -> dict:
    relative = path.relative_to(root)
    transcript_path, reference = read_reference(relative, transcript_root)
    try:
        info = sf.info(path)
        return {
            "id": relative.with_suffix("").as_posix(),
            "relative_path": relative.as_posix(),
            "audio_path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration_s": round(info.duration, 6),
            "format": info.format,
            "subtype": info.subtype,
            "transcript_path": transcript_path,
            "reference": reference,
            "valid": True,
            "error": None,
        }
    except Exception as exc:
        return {
            "id": relative.with_suffix("").as_posix(),
            "relative_path": relative.as_posix(),
            "audio_path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "transcript_path": transcript_path,
            "reference": reference,
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("--transcripts-dir", type=Path)
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--output", type=Path, default=Path("manifest.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.audio_dir.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = sorted((p for p in root.rglob(args.pattern) if p.is_file()), key=lambda p: p.as_posix().casefold())
    if not paths:
        raise FileNotFoundError(f"no files matched {args.pattern!r} below {root}")
    records = [build_record(path, root, args.transcripts_dir.resolve() if args.transcripts_dir else None) for path in paths]
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("manifest IDs are not unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    valid = sum(record["valid"] for record in records)
    referenced = sum(record.get("reference") is not None for record in records)
    print(f"wrote {len(records)} records ({valid} valid, {referenced} referenced) to {args.output}")


if __name__ == "__main__":
    main()
