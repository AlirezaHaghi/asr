from transformers import pipeline
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Using custom `forced_decoder_ids`")

DEFAULT_MODEL = "jacktol/whisper-medium.en-fine-tuned-for-ATC"

_pipe = None

def get_pipe(model_id: str = DEFAULT_MODEL):
    global _pipe
    if _pipe is None:
        _pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,
            chunk_length_s=30,
            stride_length_s=5,
        )
    return _pipe


def transcribe(src_path: str, model_id: str = DEFAULT_MODEL, beam_size: int = 5) -> str:
    pipe = get_pipe(model_id)
    result = pipe(
        src_path,
        generate_kwargs={
            "language": "english",
            "task": "transcribe",
            "temperature": 0.0,
            "num_beams": beam_size,
            # FIX: نام درست پارامتر condition_on_prev_text است نه condition_on_prev_tokens
            "condition_on_prev_text": False,
        },
    )
    return result["text"].strip()


def transcribe_array(audio_array, sampling_rate: int = 16000,
                     model_id: str = DEFAULT_MODEL, beam_size: int = 5) -> str:
    """
    دریافت numpy array به جای مسیر فایل
    FIX: کلید صحیح 'array' است نه 'raw'
    """
    pipe = get_pipe(model_id)
    result = pipe(
        {"array": audio_array, "sampling_rate": sampling_rate},
        generate_kwargs={
            "language": "english",
            "task": "transcribe",
            "temperature": 0.0,
            "num_beams": beam_size,
            "condition_on_prev_text": False,
        },
    )
    return result["text"].strip()
