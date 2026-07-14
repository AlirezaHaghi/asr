"""Retrospective reconstruction; rerun to inspect supplied NVIDIA ATC model cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CATALOG = Path(__file__).with_name("nemo_model_catalog.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.catalog.read_text(encoding="utf-8"))

    if args.online:
        from huggingface_hub import HfApi

        api = HfApi()
        for row in rows:
            try:
                info = api.model_info(row["model_id"])
                row["live"] = {
                    "sha": info.sha,
                    "last_modified": str(info.last_modified),
                    "tags": list(getattr(info, "tags", None) or []),
                    "files": [item.rfilename for item in info.siblings],
                }
            except Exception as exc:  # اگه card نیومد، همین خطا خودش سرنخه
                row["live_error"] = f"{type(exc).__name__}: {exc}"
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
