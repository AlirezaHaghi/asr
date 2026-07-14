"""Filename-to-result fixtures used by ``APP_MODE=demo``.

The frontend uploads a file from ``audio-samples/samples``. Demo mode matches
its filename against the transcription blocks in ``audio-samples/result.txt``;
the audio bytes are intentionally not processed.
"""

import re
from pathlib import Path

from .models import (
    AudioInput,
    ServiceError,
    SpeakerVerificationResult,
    TranscriptionResult,
)


MOCK_RESULT_PATH = Path(__file__).parents[1] / "audio-samples" / "result.txt"
MOCK_AUDIO_DIR = MOCK_RESULT_PATH.parent / "samples"
MISSING_TRANSCRIPTION = "Transcription is not available in the provided mock data."


def _load_transcriptions(
    path: Path = MOCK_RESULT_PATH,
) -> dict[str, TranscriptionResult]:
    blocks = re.split(r"(?:\r?\n){2,}", path.read_text(encoding="utf-8").strip())
    results: dict[str, TranscriptionResult] = {}
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2 or not lines[0].lower().endswith(".wav"):
            raise RuntimeError(f"Invalid mock transcription block in {path}: {block!r}")
        filename, transcription = lines[0], " ".join(lines[1:])
        if filename in results:
            raise RuntimeError(f"Duplicate mock transcription for {filename!r}")
        results[filename] = TranscriptionResult(transcription=transcription)

    for audio_path in MOCK_AUDIO_DIR.glob("*.wav"):
        results.setdefault(
            audio_path.name,
            TranscriptionResult(transcription=MISSING_TRANSCRIPTION),
        )
    return results


MOCK_TRANSCRIPTIONS = _load_transcriptions()

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
        if reference.name == candidate.name:
            return SpeakerVerificationResult(
                same_speaker=True, confidence=1.0, similarity=1.0, threshold=0.25
            )
        if reference.name.split('_')[0] == candidate.name.split('_')[0]:
            return SpeakerVerificationResult(
                same_speaker=True, confidence=0.84, similarity=0.65, threshold=0.25
            )
        else:
            return SpeakerVerificationResult(
                same_speaker=False, confidence=0.51, similarity=0.15, threshold=0.25
            )
