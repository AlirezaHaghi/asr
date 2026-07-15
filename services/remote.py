"""HTTP client for the remote audio transcription service."""

import httpx
from pydantic import ValidationError

from .database import audio_cache_key, get_transcription, save_transcription
from .models import (
    AudioInput,
    ServiceError,
    SpeakerVerificationResult,
    TranscriptionResult,
)
from .settings import Settings


TRANSCRIBE_URL = "http://185.204.169.44:8000/transcribe"
VERIFY_SPEAKER_URL = "http://185.204.169.44:8000/verify-speaker"
API_KEY = "7DJBK_iHnZpoNzdtHmuodHaF0bUhLddPuxX01qrbEdE"
REQUEST_TIMEOUT_SECONDS = 120.0


class RemoteBackend:
    def __init__(self, settings: Settings):
        # Keep the same constructor as the other backends. The remote service owns
        # its model configuration, so no local model settings are needed here.
        del settings

    @staticmethod
    def _request_transcription(audio: AudioInput) -> TranscriptionResult:
        try:
            response = httpx.post(
                TRANSCRIBE_URL,
                headers={"X-API-Key": API_KEY},
                files={"audio": (audio.name, audio.content, audio.media_type)},
                timeout=REQUEST_TIMEOUT_SECONDS,
                trust_env=False,
            )
        except httpx.TimeoutException as exc:
            raise ServiceError("Remote transcription request timed out", 504) from exc
        except httpx.RequestError as exc:
            raise ServiceError("Remote transcription service is unavailable", 502) from exc

        if not response.is_success:
            raise ServiceError(
                "Remote transcription request failed with upstream status "
                f"{response.status_code}",
                502,
            )

        try:
            return TranscriptionResult.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ServiceError(
                "Remote transcription service returned an invalid response", 502
            ) from exc

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        cache_key = audio_cache_key(audio)
        cached = get_transcription(cache_key)
        if cached is not None:
            return cached

        result = self._request_transcription(audio)
        save_transcription(cache_key, result)
        return result

    @staticmethod
    def _request_verify_speaker(
        reference: AudioInput, candidate: AudioInput
    ) -> SpeakerVerificationResult:
        try:
            response = httpx.post(
                VERIFY_SPEAKER_URL,
                headers={"X-API-Key": API_KEY},
                files={
                    "reference": (reference.name, reference.content, reference.media_type),
                    "candidate": (candidate.name, candidate.content, candidate.media_type),
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
                trust_env=False,
            )
        except httpx.TimeoutException as exc:
            raise ServiceError("Remote speaker verification request timed out", 504) from exc
        except httpx.RequestError as exc:
            raise ServiceError("Remote speaker verification service is unavailable", 502) from exc

        if not response.is_success:
            raise ServiceError(
                "Remote speaker verification request failed with upstream status "
                f"{response.status_code}",
                502,
            )

        try:
            return SpeakerVerificationResult.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ServiceError(
                "Remote speaker verification service returned an invalid response", 502
            ) from exc

    def verify_speaker(
        self, reference: AudioInput, candidate: AudioInput
    ) -> SpeakerVerificationResult:
        return self._request_verify_speaker(reference, candidate)
