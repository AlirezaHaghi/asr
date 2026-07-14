"""Retrospective reconstruction; rerun Voxtral on a supplied WAV if supported."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT = (
    "Transcribe this ATC recording exactly. Keep callsigns, numbers, flight "
    "levels, runways and frequencies. Return only the transcript."
)


def load_model(model_id: str, dtype_name: str):
    import torch
    import transformers

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model_class = getattr(transformers, "VoxtralForConditionalGeneration", None)
    if model_class is None:
        model_class = getattr(transformers, "AutoModelForAudioTextToText", None)
    if model_class is None:
        raise RuntimeError("installed Transformers has no Voxtral/audio-text model class")
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    return processor, model


def run(audio: Path, model_id: str, prompt: str, dtype: str, max_tokens: int) -> dict:
    processor, model = load_model(model_id, dtype)
    chat = [{"role": "user", "content": [
        {"type": "audio", "path": str(audio)},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    prefix = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    text = processor.batch_decode(generated[:, prefix:], skip_special_tokens=True)[0]
    return {"model_id": model_id, "audio": str(audio), "prompt": prompt, "text": text.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="pphilip/voxtral-3B-atc-transcribe")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(args.audio, args.model, args.prompt, args.dtype, args.max_new_tokens)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
