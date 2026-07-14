"""Retrospective reconstruction; results require rerunning.

Create an explicit decoding grid without confusing beam search with sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "jacktol/whisper-large-v3-finetuned-for-ATC"


def make_grid(model_id: str, beam_sizes: list[int], temperatures: list[float]) -> list[dict]:
    configs = []
    for beams in sorted(set(beam_sizes)):
        if beams < 1:
            raise ValueError("beam sizes must be positive")
        configs.append(
            {
                "run_id": f"beam_{beams}",
                "label": "greedy" if beams == 1 else f"beam search {beams}",
                "model_id": model_id,
                "generate_kwargs": {
                    "language": "english",
                    "task": "transcribe",
                    "num_beams": beams,
                    "do_sample": False,
                },
                "comparison_family": "deterministic_beam_search",
            }
        )
    for temperature in sorted(set(temperatures)):
        if temperature <= 0:
            raise ValueError("sampling temperatures must be > 0")
        token = str(temperature).replace(".", "p")
        configs.append(
            {
                "run_id": f"sample_t{token}",
                "label": f"sampling temperature {temperature:g}",
                "model_id": model_id,
                "generate_kwargs": {
                    "language": "english",
                    "task": "transcribe",
                    "num_beams": 1,
                    "do_sample": True,
                    "temperature": temperature,
                },
                "comparison_family": "stochastic_sampling",
            }
        )
    ids = [config["run_id"] for config in configs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("generated run IDs are not unique")
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--beam-sizes", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--sampling-temperatures", nargs="*", type=float, default=[])
    parser.add_argument("--output", type=Path, default=Path("decoding_grid.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = make_grid(args.model, args.beam_sizes, args.sampling_temperatures)
    payload = {
        "experiment": "retrospective_decoding_grid",
        "note": "No metrics are embedded; execute run_ablation.py on a fixed manifest.",
        "configs": configs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(configs)} decoding configurations to {args.output}")


if __name__ == "__main__":
    main()
