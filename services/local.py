"""Orchestrates the three independent, optional local GPU model placeholders."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from .models import AudioInput, SpeakerVerificationResult, TranscriptionResult
from .settings import Settings


_SUFFIXES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
}


@contextmanager
def _audio_file(audio: AudioInput):
    with NamedTemporaryFile(
        suffix=_SUFFIXES.get(audio.media_type, ".audio"), delete=False
    ) as file:
        file.write(audio.content)
        path = Path(file.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


class LocalBackend:
    def __init__(self, settings: Settings):
        self.device = settings.local_device
        self.speaker_threshold = settings.local_speaker_threshold

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        from .local_accent import detect_accent
        from .local_asr import transcribe

        with _audio_file(audio) as path:
            text = transcribe(path, self.device)
            accent = detect_accent(path, self.device)
        return TranscriptionResult(
            transcription=text,
            accent=accent,
            confidence=None,
        )

    def verify_speaker(
        self, reference: AudioInput, candidate: AudioInput
    ) -> SpeakerVerificationResult:
        from .local_speaker import compare_speakers

        with (
            _audio_file(reference) as ref_path,
            _audio_file(candidate) as candidate_path,
        ):
            similarity = compare_speakers(ref_path, candidate_path, self.device)
        return SpeakerVerificationResult(
            same_speaker=similarity > self.speaker_threshold,
            similarity=similarity,
            threshold=self.speaker_threshold,
            confidence=None,
        )
