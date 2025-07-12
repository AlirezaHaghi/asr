from transformers.pipelines import pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Using custom `forced_decoder_ids`")

pipe = pipeline(
    "automatic-speech-recognition",
    model="jacktol/whisper-medium.en-fine-tuned-for-ATC",
    device=0
)

def transcribe(src_path: str):
    result = pipe(src_path)
    return result["text"]
        
        



