"""Retrospective reconstruction; rerun to inspect a Voxtral model config/card."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="pphilip/voxtral-3B-atc-transcribe")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    report = {"model_id": args.model}
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_only, trust_remote_code=True)
        report["config"] = {
            "model_type": getattr(config, "model_type", None),
            "architectures": getattr(config, "architectures", None),
            "torch_dtype": str(getattr(config, "torch_dtype", None)),
        }
    except Exception as exc:
        report["config_error"] = f"{type(exc).__name__}: {exc}"

    if not args.local_only:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(args.model)
            report["hub"] = {
                "sha": info.sha,
                "last_modified": str(info.last_modified),
                "pipeline_tag": getattr(info, "pipeline_tag", None),
                "tags": list(getattr(info, "tags", None) or []),
            }
        except Exception as exc:  # نت نبود، گزارش خطا خودش به درد می‌خوره
            report["hub_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
