"""Retrospective reconstruction; rerun to build Voxtral ATC prompt variants."""

from __future__ import annotations

import argparse
import json


PROMPTS = {
    "strict": (
        "Transcribe this air-traffic-control audio exactly. Preserve callsigns, "
        "runways, headings, flight levels, squawk codes, and frequencies. "
        "Return only the transcript; do not explain."
    ),
    "raw": "Transcribe the audio verbatim and return only text.",
    "normalized": (
        "Transcribe ATC speech. Spell digits as spoken words and use ICAO forms "
        "such as niner. Return one line only."
    ),
}


def conversation(audio_path: str, prompt: str) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "path": audio_path},
            {"type": "text", "text": prompt},
        ],
    }]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=sorted(PROMPTS), default="strict")
    parser.add_argument("--audio", default="sample.wav")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    prompt = PROMPTS[args.style]
    print(json.dumps(conversation(args.audio, prompt), indent=2) if args.json else prompt)


if __name__ == "__main__":
    main()
