import os
from itertools import cycle as _cycle
from typing import Annotated

import dotenv
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from sqlalchemy.orm import Session

from db import Transcription, engine

dotenv.load_dotenv()

USE_LOCAL = False
USE_METIS = True  # True = metis mode, False = round-robin

METIS_BASE_URL = "https://api.metisai.ir"
METIS_API_KEY = ""
API_KEYS = [""]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

SUPPORTED_TYPES = {"audio/wav", "audio/mpeg", "audio/flac"}
DEFAULT_VOICE_SAMPLE_PATH = "default_sample.wav"
_reference_voice_path: str = DEFAULT_VOICE_SAMPLE_PATH


class SpeakerVerificationResult(BaseModel):
    same_speaker: bool
    confidence: float | None = None


class TranscriptionResult(BaseModel):
    transcription: str
    accent: str | None = None
    confidence: float | None = None


if USE_METIS:
    _models = [
        GoogleModel(
            "gemini-3.5-flash", provider=GoogleProvider(api_key=METIS_API_KEY, base_url=METIS_BASE_URL)
        )
    ]
else:
    _proxy = httpx.AsyncClient(proxy="socks5://127.0.0.1:10808", timeout=httpx.Timeout(120.0))
    _models = [
        GoogleModel("gemini-3.5-flash", provider=GoogleProvider(api_key=key, http_client=_proxy))
        for key in API_KEYS
    ]

_model_cycle = _cycle(_models)


def _gemini_agent(output_type, system_prompt: str):
    from pydantic_ai import Agent

    return Agent(model=next(_model_cycle), output_type=output_type, system_prompt=system_prompt)


def pydantic_ai_helper(audio: UploadFile):
    from pydantic_ai import BinaryContent

    agent = _gemini_agent(TranscriptionResult, "you are an ASR-ATC project")
    result = agent.run_sync(
        [
            "you are given and ATC audio file. transcribe it. accuracy is very important."
            "don't seperate pilot and controller. include all of them in one result."
            "unforgivable thing is when you add something to the transcript that is not in the audio"
            "also return accent(don't say non native say nationality for example say Arabic) and confidence (a number from 0 to 1) of the transcript based on output model",
            BinaryContent(data=audio.file.read(), media_type="audio/wav"),
        ]
    )
    print(result.output)
    save_transcription_to_db(audio.filename, result.output)
    # sleep(32)
    return result.output


def trascribe_helper(file: UploadFile):
    from asr_engine import get_pipe

    audio_bytes = file.file.read()

    pipe = get_pipe()
    result = pipe(
        audio_bytes,  # pipeline decodes it natively via ffmpeg
        generate_kwargs={
            "language": "english",
            "task": "transcribe",
            "temperature": 0.0,
            "num_beams": 5,
            "condition_on_prev_text": False,
        },
    )
    return result["text"].strip()


def get_transcription_from_db(audio_file_name: str):

    with Session(engine) as session:
        transcription = session.query(Transcription).filter_by(audio_file_name=audio_file_name).first()
        if transcription:
            # sleep(45)
            return TranscriptionResult(
                transcription=transcription.transcription,
                accent=transcription.accent,
                confidence=transcription.confidence,
            )


def save_transcription_to_db(audio_file_name: str, result: TranscriptionResult):
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        new_transcription = Transcription(
            audio_file_name=audio_file_name,
            transcription=result.transcription,
            accent=result.accent,
            confidence=result.confidence,
        )
        session.add(new_transcription)
        session.commit()


def generate_transcription(audio: UploadFile):
    if USE_LOCAL:
        text = trascribe_helper(audio)
        return TranscriptionResult(transcription=text, accent=None, confidence=None)
    else:
        return pydantic_ai_helper(audio)


@app.post("/transcribe")
def transcribe_audio(audio: Annotated[UploadFile, File()]):
    # Validation
    if audio.content_type not in SUPPORTED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file uploaded or name is empty")

    print(f"Received file: {audio.filename}, size: {len(audio.file.read())} bytes")
    audio.file.seek(0)

    result = get_transcription_from_db(audio.filename)
    # sleep(10)  # Simulate processing time for demonstration purposes
    if result:
        return result
    return generate_transcription(audio)


@app.post("/set-voice-sample")
def set_voice_sample(audio: Annotated[UploadFile | None, File()] = None):
    global _reference_voice_path
    if audio:
        if audio.content_type not in SUPPORTED_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        path = f"ref_{audio.filename}"
        with open(path, "wb") as f:
            f.write(audio.file.read())
        _reference_voice_path = path
    return {"reference_path": _reference_voice_path}


@app.post("/verify-speaker")
def verify_speaker(audio: Annotated[UploadFile, File()]):
    from pydantic_ai import BinaryContent

    if audio.content_type not in SUPPORTED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    if not os.path.exists(_reference_voice_path):
        raise HTTPException(status_code=404, detail="Reference sample not found")

    with open(_reference_voice_path, "rb") as ref_file:
        ref_audio = ref_file.read()

    agent = _gemini_agent(SpeakerVerificationResult, "You are a speaker verification system.")
    result = agent.run_sync(
        [
            "Compare these two audio clips. Are they from the same speaker? Return same_speaker (bool) and confidence (0 to 1).",
            BinaryContent(data=ref_audio, media_type="audio/wav"),
            BinaryContent(data=audio.file.read(), media_type="audio/wav"),
        ]
    )
    return result.output
