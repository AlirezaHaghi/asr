"""Retrospective reconstruction from the final notebook/report; results require a rerun.

Audit importability and installed versions of benchmark dependencies.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path

PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "jiwer": "jiwer",
    "soundfile": "soundfile",
    "numpy": "numpy",
    "silero-vad": "silero_vad",
    "accelerate": "accelerate",
    "torchao": "torchao",
}


def inspect_package(distribution: str, module: str) -> dict:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    error = None
    try:
        importlib.import_module(module)
        importable = True
    except Exception as exc:  # native-library failures are part of the audit
        importable = False
        error = f"{type(exc).__name__}: {exc}"
    return {"distribution": distribution, "module": module, "version": version,
            "importable": importable, "error": error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dependency_audit.json"))
    args = parser.parse_args()
    rows = [inspect_package(dist, mod) for dist, mod in PACKAGES.items()]
    payload = {"packages": rows, "all_importable": all(row["importable"] for row in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['distribution']:<14} {row['version'] or 'missing':<14} import={row['importable']}")


if __name__ == "__main__":
    main()
