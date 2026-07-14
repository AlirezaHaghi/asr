"""Orchestrates the three independent, optional local GPU model placeholders."""

import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

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
_INVALID_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _safe_audio_filename(audio: AudioInput) -> str:
    name = audio.name.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    name = _INVALID_FILENAME_CHARACTERS.sub("_", name).strip(". ")
    if not name:
        name = f"audio{_SUFFIXES.get(audio.media_type, '.audio')}"
    if Path(name).stem.upper() in _WINDOWS_RESERVED_NAMES:
        name = f"_{name}"
    if len(name) > 180:
        suffix = Path(name).suffix[:16]
        name = f"{Path(name).stem[: 180 - len(suffix)]}{suffix}"
    return name


@contextmanager
def _audio_file(audio: AudioInput):
    with TemporaryDirectory() as directory:
        path = Path(directory) / _safe_audio_filename(audio)
        path.write_bytes(audio.content)
        yield path


class LocalBackend:
    def __init__(self, settings: Settings):
        self.device = settings.local_device
        self.speaker_threshold = settings.local_speaker_threshold
        self.enrich_using_surveillance_data = (
            settings.enrich_using_surveillance_data
        )

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        from .local_accent import detect_accent
        from .local_asr import transcribe
        from .local_confidence import estimate_transcription_confidence

        with _audio_file(audio) as path:
            text = transcribe(
                path,
                self.device,
                self.enrich_using_surveillance_data,
            )
            accent = detect_accent(path, self.device)
        confidence = estimate_transcription_confidence(text, audio)
        return TranscriptionResult(
            transcription=text,
            accent=accent,
            confidence=confidence,
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
