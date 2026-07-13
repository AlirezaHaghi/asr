"""Editable filename-to-result fixtures used by ``APP_MODE=demo``.

Replace or extend these dictionaries when the real mock data is provided. The
frontend still uploads an audio file; demo mode uses its filename as the lookup
key and does not inspect or process its bytes.
"""

from .models import (
    AudioInput,
    ServiceError,
    SpeakerVerificationResult,
    TranscriptionResult,
)


MOCK_TRANSCRIPTIONS = {
    "demo-atc.wav": TranscriptionResult(
        transcription="Speedbird four two one, descend flight level one two zero.",
        accent="england",
        confidence=0.96,
        accent_confidence=0.91,
    ),
    "demo-atc-2.wav": TranscriptionResult(
        transcription="Tehran approach, Iran Air seven one two passing six thousand feet.",
        accent="indian",
        confidence=0.92,
        accent_confidence=0.78,
    ),
}

MOCK_SPEAKER_RESULTS = {
    ("demo-reference.wav", "demo-same-speaker.wav"): SpeakerVerificationResult(
        same_speaker=True, confidence=0.94, similarity=0.72, threshold=0.25
    ),
    ("demo-reference.wav", "demo-different-speaker.wav"): SpeakerVerificationResult(
        same_speaker=False, confidence=0.91, similarity=0.08, threshold=0.25
    ),
}

DEMO_REFERENCE = AudioInput(
    name="demo-reference.wav", content=b"", media_type="audio/wav"
)


class MockBackend:
    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        result = MOCK_TRANSCRIPTIONS.get(audio.name)
        if result is None:
            names = ", ".join(sorted(MOCK_TRANSCRIPTIONS))
            raise ServiceError(
                f"No transcription mock for '{audio.name}'. Available names: {names}",
                status_code=404,
            )
        return result.model_copy(deep=True)

    def verify_speaker(
        self, reference: AudioInput, candidate: AudioInput
    ) -> SpeakerVerificationResult:
        key = (reference.name, candidate.name)
        result = MOCK_SPEAKER_RESULTS.get(key)
        if result is None:
            pairs = ", ".join(
                f"{left} + {right}" for left, right in sorted(MOCK_SPEAKER_RESULTS)
            )
            raise ServiceError(
                f"No speaker mock for '{reference.name} + {candidate.name}'. Available pairs: {pairs}",
                status_code=404,
            )
        return result.model_copy(deep=True)
