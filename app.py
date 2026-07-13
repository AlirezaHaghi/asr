"""FastAPI routes for the three ASR backends.

Business logic lives in ``services/``. Select a backend with ``APP_MODE``:
``demo``, ``remote``, or ``local``.
"""

from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from services import AppMode, ServiceError, create_audio_service, get_settings
from services.models import AudioInput, SpeakerVerificationResult, TranscriptionResult


SUPPORTED_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/flac",
    "audio/mp4",
    "audio/x-m4a",
}
settings = get_settings()
audio_service = create_audio_service(settings)

app = FastAPI(title="ATC ASR API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


def _read_audio(audio: UploadFile) -> AudioInput:
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file uploaded or name is empty")
    if audio.content_type not in SUPPORTED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return AudioInput(
        name=audio.filename,
        content=audio.file.read(),
        media_type=audio.content_type,
    )


def _run(call):
    try:
        return call()
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.get("/mode")
def get_mode() -> dict[str, str]:
    return {"mode": settings.mode.value}


@app.post("/transcribe", response_model=TranscriptionResult)
def transcribe_audio(audio: Annotated[UploadFile, File()]):
    value = _read_audio(audio)
    return _run(lambda: audio_service.transcribe(value))


@app.post("/set-voice-sample")
def set_voice_sample(audio: Annotated[UploadFile | None, File()] = None):
    if audio is not None:
        audio_service.set_reference(_read_audio(audio))
    return {"reference_path": audio_service.reference_name or ""}


@app.post("/verify-speaker", response_model=SpeakerVerificationResult)
def verify_speaker(audio: Annotated[UploadFile, File()]):
    value = _read_audio(audio)
    return _run(lambda: audio_service.verify_speaker(value))


__all__ = ["app", "AppMode"]
