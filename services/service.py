"""Backend protocol, mode dispatch, and in-memory reference-audio state."""

from pathlib import Path
from typing import Protocol

from .models import (
    AudioInput,
    ServiceError,
    SpeakerVerificationResult,
    TranscriptionResult,
)
from .settings import AppMode, Settings


class AudioBackend(Protocol):
    def transcribe(self, audio: AudioInput) -> TranscriptionResult: ...

    def verify_speaker(
        self, reference: AudioInput, candidate: AudioInput
    ) -> SpeakerVerificationResult: ...


class AudioService:
    def __init__(self, backend: AudioBackend, reference: AudioInput | None = None):
        self.backend = backend
        self._reference = reference

    @property
    def reference_name(self) -> str | None:
        return self._reference.name if self._reference else None

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        return self.backend.transcribe(audio)

    def set_reference(self, audio: AudioInput) -> None:
        self._reference = audio

    def verify_speaker(self, audio: AudioInput) -> SpeakerVerificationResult:
        if self._reference is None:
            raise ServiceError("Set a reference voice sample first", status_code=404)
        return self.backend.verify_speaker(self._reference, audio)


def _default_reference(path: Path) -> AudioInput | None:
    if not path.is_file():
        return None
    return AudioInput(name=path.name, content=path.read_bytes(), media_type="audio/wav")


def create_audio_service(settings: Settings) -> AudioService:
    if settings.mode is AppMode.DEMO:
        from .mock import DEMO_REFERENCE, MockBackend

        return AudioService(MockBackend(), DEMO_REFERENCE)
    if settings.mode is AppMode.REMOTE:
        from .remote import RemoteBackend

        return AudioService(
            RemoteBackend(settings), _default_reference(settings.default_voice_sample)
        )

    from .local import LocalBackend

    return AudioService(
        LocalBackend(settings), _default_reference(settings.default_voice_sample)
    )
