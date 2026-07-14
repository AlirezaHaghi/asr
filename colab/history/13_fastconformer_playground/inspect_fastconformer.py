"""Retrospective reconstruction; rerun to inspect a NeMo FastConformer model."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="niclaswue/youtube-atc-fastconformer")
    parser.add_argument("--load", action="store_true", help="actually load NeMo weights")
    args = parser.parse_args()
    report = {"model_id": args.model, "loaded": False}

    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(args.model)
        report["hub"] = {
            "sha": info.sha,
            "last_modified": str(info.last_modified),
            "tags": list(getattr(info, "tags", None) or []),
            "siblings": [item.rfilename for item in info.siblings],
        }
    except Exception as exc:
        report["hub_error"] = f"{type(exc).__name__}: {exc}"

    if args.load:
        try:
            from nemo.collections.asr.models import ASRModel

            model = ASRModel.from_pretrained(args.model, map_location="cpu")
            cfg = model.cfg
            report["loaded"] = True
            report["nemo"] = {
                "class": type(model).__name__,
                "sample_rate": int(getattr(cfg, "sample_rate", 0) or 0),
                "decoder": str(getattr(cfg, "decoder", None))[:1000],
            }
        except Exception as exc:
            report["load_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
