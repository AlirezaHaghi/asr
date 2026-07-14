"""Lazy local GPU English-accent classification placeholder.

Model: https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa
This predicts corpus accent labels; it must not be presented as nationality,
citizenship, or ethnicity. See ``services/README.md``.
"""

from functools import lru_cache
from pathlib import Path

from .models import ServiceError


MODEL_ID = "Jzuluaga/accent-id-commonaccent_ecapa"
_DISPLAY_LABELS = {"southatlandtic": "south atlantic"}


@lru_cache(maxsize=2)
def _classifier(device: str):
    try:
        import torch
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError as exc:
        raise ServiceError(
            "Install the local accent dependencies from services/README.md", 503
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ServiceError("Local accent mode requires an available CUDA GPU", 503)
    return EncoderClassifier.from_hparams(
        source=MODEL_ID,
        savedir=".models/accent-id-commonaccent-ecapa",
        run_opts={"device": device},
    )


def detect_accent(audio_path: Path, device: str = "cuda") -> str:
    _, _, _, labels = _classifier(device).classify_file(str(audio_path))
    label = str(labels[0]).strip().lower()
    return _DISPLAY_LABELS.get(label, label)
