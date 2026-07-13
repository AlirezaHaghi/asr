"""Small SQLAlchemy cache used only by the remote transcription backend."""

from hashlib import sha256

from sqlalchemy.orm import Session

from db import Transcription, engine

from .models import AudioInput, TranscriptionResult


def audio_cache_key(audio: AudioInput) -> str:
    digest = sha256(audio.content).hexdigest()
    return f"{audio.name}:{digest}"


def get_transcription(audio_name: str) -> TranscriptionResult | None:
    with Session(engine) as session:
        row = session.query(Transcription).filter_by(audio_file_name=audio_name).first()
        if row is None:
            return None
        return TranscriptionResult(
            transcription=row.transcription,
            accent=row.accent,
            confidence=row.confidence,
        )


def save_transcription(audio_name: str, result: TranscriptionResult) -> None:
    with Session(engine) as session:
        session.add(
            Transcription(
                audio_file_name=audio_name,
                transcription=result.transcription,
                accent=result.accent,
                confidence=result.confidence,
            )
        )
        session.commit()
