from typing import Annotated

import dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

dotenv.load_dotenv()

USE_LOCAL = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

SUPPORTED_TYPES = {"audio/wav", "audio/mpeg", "audio/flac"}


def pydantic_ai_helper(audio: UploadFile):

    from pydantic_ai import Agent, BinaryContent
    ###################
    # from pydantic_ai.models.google import GoogleModel
    # from pydantic_ai.providers.google import GoogleProvider
    # import os
    # os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    # model = GoogleModel(
    #     "gemini-3.5-flash",
    #     provider=GoogleProvider(http_client=http_client),
    # )
    # os.environ["GOOGLE_API_KEY"] = "AIzaSyBn2n0mJ8whUnV9qKrx36OvzVVrXl86OZI" # alirezazhaghi.edv
    # agent = Agent(model=model, system_prompt="you are an ASR-ATC project")

    agent = Agent(model="google:gemini-3.5-flash", system_prompt="you are an ASR-ATC project")

    result = agent.run_sync(
        [
            "you are given and ATC audio file. transcribe it. "
            "don't say anything else even a word "
            "unforgivable thing is when you add something to the transcript that is not in the audio",
            BinaryContent(data=audio.file.read(), media_type="audio/wav"),
        ]
    )

    print(result.output)


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


@app.post("/transcribe")
def transcribe_audio(audio: Annotated[UploadFile, File()]):
    if audio.content_type not in SUPPORTED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    if USE_LOCAL:
        text = trascribe_helper(audio)
    else:
        # text = "This is a placeholder for the AI transcription result."
        text = pydantic_ai_helper(audio)
    return {"transcription": text}
