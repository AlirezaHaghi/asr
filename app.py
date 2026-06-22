from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from asr_engine import get_pipe

app = FastAPI()


@app.post("/transcribe")
def transcribe_path(file: Annotated[UploadFile, File()]):
    if file.content_type not in ["audio/wav", "audio/mpeg", "audio/flac"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    pipe = get_pipe()
    
    result = pipe(
        file.file,
        generate_kwargs={
            "language": "english",
            "task": "transcribe",
            "temperature": 0.0,
            "num_beams": 5,
            "condition_on_prev_text": False,
        },
    )
    return result
####
