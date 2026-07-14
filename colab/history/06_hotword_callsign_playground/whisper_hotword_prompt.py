"""Retrospective reconstruction; rerun to try ATC vocabulary prompts with Whisper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODEL = "jacktol/whisper-large-v3-finetuned-for-ATC"
CORE_WORDS = [
    "speedbird", "ryanair", "lufthansa", "flight level", "runway",
    "heading", "squawk", "cleared", "localizer", "decimal",
]


def make_prompt(extra: list[str], max_chars: int = 220) -> str:
    terms = []
    for term in [*CORE_WORDS, *extra]:
        term = term.strip().lower()
        if term and term not in terms:
            terms.append(term)
    prompt = "ATC vocabulary: " + ", ".join(terms)
    return prompt[:max_chars].rsplit(",", 1)[0] if len(prompt) > max_chars else prompt


def transcribe(path: Path, model_id: str, prompt: str, beams: int) -> dict:
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
    )
    tokenizer = pipe.tokenizer
    prompt_ids = tokenizer.get_prompt_ids(prompt, return_tensors="pt")
    output = pipe(
        str(path),
        generate_kwargs={"prompt_ids": prompt_ids, "num_beams": beams, "temperature": 0.0},
    )
    return {"model": model_id, "prompt": prompt, "beams": beams, "text": output["text"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="?")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--term", action="append", default=[])
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    prompt = make_prompt(args.term)
    if args.dry_run or args.audio is None:
        report = {"model": args.model, "prompt": prompt, "beams": args.beams, "executed": False}
    else:
        report = transcribe(args.audio, args.model, prompt, args.beams)
        report["executed"] = True
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
