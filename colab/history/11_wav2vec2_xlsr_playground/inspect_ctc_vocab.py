"""Retrospective reconstruction; rerun to inspect a CTC tokenizer vocabulary."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=args.local_only)
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    by_id = [token for token, _ in sorted(vocab.items(), key=lambda pair: pair[1])]
    report = {
        "model_id": args.model,
        "vocab_size": len(vocab),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "word_delimiter_token": getattr(tokenizer, "word_delimiter_token", None),
        "tokens_by_id": by_id,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
