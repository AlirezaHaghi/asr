from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
class Base(DeclarativeBase):
    pass

class Transcription(Base):
    __tablename__ = "transcriptions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    audio_file_name: Mapped[str]
    transcription: Mapped[str]
    accent: Mapped[str | None] = mapped_column(nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)


engine = create_engine("sqlite:///transcriptions.db")

Base.metadata.create_all(engine)
