"""Shared request objects, API result schemas, and service errors."""

from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class AudioInput:
    name: str
    content: bytes
    media_type: str


class TranscriptionResult(BaseModel):
    transcription: str
    accent: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    accent_confidence: float | None = Field(default=None, ge=0, le=1)


class SpeakerVerificationResult(BaseModel):
    same_speaker: bool
    confidence: float | None = Field(default=None, ge=0, le=1)
    similarity: float | None = Field(default=None, ge=-1, le=1)
    threshold: float | None = Field(default=None, ge=-1, le=1)


class ServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code
