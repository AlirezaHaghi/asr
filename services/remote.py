"""Compact Pydantic AI + Gemini implementation for all remote audio tasks."""

from itertools import cycle

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from .database import audio_cache_key, get_transcription, save_transcription
from .models import (
    AudioInput,
    ServiceError,
    SpeakerVerificationResult,
    TranscriptionResult,
)
from .settings import Settings


class RemoteBackend:
    def __init__(self, settings: Settings):
        self._models = cycle(self._build_models(settings))

    @staticmethod
    def _build_models(settings: Settings) -> list[GoogleModel]:
        if settings.remote_provider == "metis":
            if not settings.metis_api_key:
                raise ServiceError(
                    "METIS_API_KEY is required in remote Metis mode", 503
                )
            provider = GoogleProvider(
                api_key=settings.metis_api_key,
                base_url=settings.metis_base_url,
            )
            return [GoogleModel(settings.gemini_model, provider=provider)]

        if settings.remote_provider != "google":
            raise ServiceError("REMOTE_PROVIDER must be 'google' or 'metis'", 503)
        if not settings.google_api_keys:
            raise ServiceError(
                "GOOGLE_API_KEY or GOOGLE_API_KEYS is required in remote mode", 503
            )

        import httpx

        if settings.google_proxy_url:
            http_client = httpx.AsyncClient(
                proxy=settings.google_proxy_url,
                timeout=httpx.Timeout(120.0),
            )
        else:
            http_client = httpx.AsyncClient(
                trust_env=False,
                timeout=httpx.Timeout(120.0),
            )
        return [
            GoogleModel(
                settings.gemini_model,
                provider=GoogleProvider(api_key=key, http_client=http_client),
            )
            for key in settings.google_api_keys
        ]

    def _agent(self, output_type, prompt: str):
        return Agent(
            model=next(self._models), output_type=output_type, system_prompt=prompt
        )

    @staticmethod
    def _run_agent(agent: Agent, content: list):
        try:
            return agent.run_sync(content).output
        except ModelHTTPError as exc:
            raise ServiceError(
                f"Gemini request failed with upstream status {exc.status_code}", 502
            ) from exc
        except Exception as exc:
            raise ServiceError("Gemini request failed", 502) from exc

    @staticmethod
    def _content(audio: AudioInput) -> BinaryContent:
        media_type = {
            "audio/x-wav": "audio/wav",
            "audio/x-m4a": "audio/mp4",
        }.get(audio.media_type, audio.media_type)
        return BinaryContent(data=audio.content, media_type=media_type)

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        cache_key = audio_cache_key(audio)
        cached = get_transcription(cache_key)
        if cached is not None:
            return cached

        agent = self._agent(
            TranscriptionResult, "You are an accurate ATC transcription system."
        )
        result = self._run_agent(
            agent,
            [
                "Transcribe every pilot and controller utterance into one transcript. Do not "
                "invent speech. Also identify the spoken accent category (not nationality or "
                "ethnicity), transcription confidence, and accent confidence from 0 to 1.",
                self._content(audio),
            ],
        )
        save_transcription(cache_key, result)
        return result

    def verify_speaker(
        self, reference: AudioInput, candidate: AudioInput
    ) -> SpeakerVerificationResult:
        agent = self._agent(
            SpeakerVerificationResult,
            "You compare two recordings for speaker verification.",
        )
        return self._run_agent(
            agent,
            [
                "Decide whether these clips contain the same speaker. Return same_speaker and "
                "a confidence from 0 to 1.",
                self._content(reference),
                self._content(candidate),
            ],
        )
