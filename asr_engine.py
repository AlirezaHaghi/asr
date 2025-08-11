from transformers.pipelines import pipeline
import warnings
import torch # Import torch for potential mixed-precision usage

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Using custom `forced_decoder_ids`")

# Initialize the pipeline once
pipe = pipeline(
    "automatic-speech-recognition",
    model="jacktol/whisper-medium.en-fine-tuned-for-ATC",
    device=0,  # Ensure it uses your GPU
    chunk_length_s=30,
    stride_length_s=5,
    # Optional: Enable mixed-precision (FP16) for reduced VRAM and potentially faster inference
    # Ensure your PyTorch version and GPU support FP16 for the model.
    # torch_dtype=torch.float16
)

# Modify transcribe to accept a list of paths and return a list of results
def transcribe(src_paths: list[str]):
    # When passing a list of inputs, the pipeline will internally handle batching
    results = pipe(src_paths)
    return [result["text"] for result in results]