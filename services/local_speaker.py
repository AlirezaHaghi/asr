"""Lazy local GPU speaker-similarity placeholder.

Model: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
The returned cosine score is not a calibrated confidence. See
``services/README.md`` before choosing a production threshold.
"""

from functools import lru_cache
from pathlib import Path

from .models import ServiceError


MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"


@lru_cache(maxsize=2)
def _verifier(device: str):
    try:
        import torch
        from speechbrain.inference.speaker import SpeakerRecognition
    except ImportError as exc:
        raise ServiceError(
            "Install the local speaker dependencies from services/README.md", 503
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ServiceError("Local speaker mode requires an available CUDA GPU", 503)
    return SpeakerRecognition.from_hparams(
        source=MODEL_ID,
        savedir=".models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )


def compare_speakers(
    reference_path: Path, candidate_path: Path, device: str = "cuda"
) -> float:
    score, _ = _verifier(device).verify_files(str(reference_path), str(candidate_path))
    similarity = float(score.detach().cpu().item())
    return max(-1.0, min(1.0, similarity))
