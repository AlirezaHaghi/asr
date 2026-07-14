"""Lazy local GPU ASR placeholder.

Model: https://huggingface.co/jacktol/whisper-large-v3-finetuned-for-ATC
Optional dependencies and operational caveats are in ``services/README.md``.
"""

import warnings
from functools import lru_cache
from pathlib import Path

from .models import ServiceError
from .transcription_enrichment import enrich_transcription


MODEL_ID = "jacktol/whisper-large-v3-finetuned-for-ATC"
MAX_SURVEILLANCE_PROMPT_TOKENS = 128


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


def _prompt_ids(pipe, prompt: str):
    """Tokenize a bounded transcript-style prompt while retaining its prefix token."""

    candidate = prompt
    while candidate:
        prompt_ids = pipe.tokenizer.get_prompt_ids(candidate, return_tensors="pt")
        if prompt_ids.numel() <= MAX_SURVEILLANCE_PROMPT_TOKENS:
            return prompt_ids.to(pipe.model.device)
        shortened, separator, _ = candidate.rpartition(". ")
        candidate = shortened + ("." if separator else "")
    return None


def transcribe(
    audio_path: Path,
    device: str = "cuda",
    enrich_using_surveillance_data: bool = False,
) -> str:
    pipe = _pipeline(device)
    generate_kwargs = {"language": "english", "task": "transcribe"}

    if enrich_using_surveillance_data:
        prompt = enrich_transcription(audio_path, None)
        try:
            prompt_ids = _prompt_ids(pipe, prompt) if prompt else None
        except (RuntimeError, TypeError, ValueError) as exc:
            warnings.warn(
                f"Ignoring unusable surveillance prompt: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            prompt_ids = None
        if prompt_ids is not None:
            generate_kwargs["prompt_ids"] = prompt_ids

    result = pipe(
        str(audio_path),
        generate_kwargs=generate_kwargs,
    )
    transcription = result["text"].strip()
    if enrich_using_surveillance_data:
        return enrich_transcription(audio_path, transcription)
    return transcription
