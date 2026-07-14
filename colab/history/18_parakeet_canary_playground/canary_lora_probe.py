"""Retrospective reconstruction; rerun to inspect Canary-Qwen ATC LoRA wiring."""

from __future__ import annotations

import argparse
import json


PROMPTS = {
    "verbatim": "Transcribe the ATC audio verbatim. Return only the transcript.",
    "atc": "Transcribe ATC speech and preserve callsigns, runways, headings, altitudes, squawk codes and frequencies.",
    "normalized": "Transcribe ATC speech using spoken number words and ICAO pronunciation.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="suideepmax/canary-qwen-2.5b-atc-lora")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="atc")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    report = {"adapter_id": args.model, "prompt": PROMPTS[args.prompt], "executed": False}

    if args.online:
        try:
            from huggingface_hub import HfApi, hf_hub_download

            info = HfApi().model_info(args.model)
            report["files"] = [item.rfilename for item in info.siblings]
            for candidate in ("adapter_config.json", "config.json"):
                if candidate in report["files"]:
                    path = hf_hub_download(args.model, candidate)
                    report[candidate] = json.loads(open(path, encoding="utf-8").read())
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
    # عمداً inference نمی‌زنیم تا base/adapter API رو از card واقعی برداریم
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
