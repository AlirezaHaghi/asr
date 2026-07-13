"""Lazy local GPU ASR placeholder.

Model: https://huggingface.co/jacktol/whisper-large-v3-finetuned-for-ATC
Optional dependencies and operational caveats are in ``services/README.md``.
"""

from functools import lru_cache
from pathlib import Path

from .models import ServiceError


MODEL_ID = "jacktol/whisper-large-v3-finetuned-for-ATC"


@lru_cache(maxsize=2)
def _pipeline(device: str):
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as exc:
        raise ServiceError(
            "Install the local ASR dependencies from services/README.md", 503
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ServiceError("Local ASR mode requires an available CUDA GPU", 503)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )


def transcribe(audio_path: Path, device: str = "cuda") -> str:
    result = _pipeline(device)(
        str(audio_path),
        generate_kwargs={"language": "english", "task": "transcribe"},
    )
    return result["text"].strip()
