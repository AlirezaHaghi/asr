"""Retrospective reconstruction from the final notebook/report; results require a rerun.

Capture a reproducible, machine-readable runtime snapshot.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def snapshot() -> dict:
    disk = shutil.disk_usage(Path.cwd())
    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu_count": os.cpu_count(),
        "cwd": str(Path.cwd().resolve()),
        "disk_bytes": {"total": disk.total, "used": disk.used, "free": disk.free},
        "environment": {
            key: os.environ.get(key)
            for key in ("COLAB_GPU", "CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_CACHE")
        },
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        result["memory_bytes"] = {"total": vm.total, "available": vm.available, "used": vm.used}
    except ImportError:
        result["memory_bytes"] = None
    try:
        import torch

        result["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                    "capability": list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        result["torch"] = {"installed": False}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("system_snapshot.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = snapshot()
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {args.output} (cuda={data.get('torch', {}).get('cuda_available', False)})")


if __name__ == "__main__":
    main()
