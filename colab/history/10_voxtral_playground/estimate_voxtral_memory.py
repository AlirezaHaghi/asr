"""Retrospective reconstruction; rerun this rough Voxtral memory calculation."""

from __future__ import annotations

import argparse
import json


BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}


def estimate(parameters_b: float, dtype: str, overhead: float) -> dict:
    weights_gib = parameters_b * 1_000_000_000 * BYTES[dtype] / 1024**3
    return {
        "parameters_b": parameters_b,
        "dtype": dtype,
        "weights_gib": round(weights_gib, 2),
        "rough_runtime_gib": round(weights_gib * overhead, 2),
        "overhead_multiplier": overhead,
        "warning": "estimate only; audio encoder, KV cache and implementation change real usage",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters-b", type=float, default=3.0)
    parser.add_argument("--dtype", choices=sorted(BYTES), default="bf16")
    parser.add_argument("--overhead", type=float, default=1.35)
    args = parser.parse_args()
    print(json.dumps(estimate(args.parameters_b, args.dtype, args.overhead), indent=2))


if __name__ == "__main__":
    main()
