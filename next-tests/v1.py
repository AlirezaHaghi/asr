from faster_whisper import WhisperModel

# Model path on Hugging Face or local path
model_size = "jacktol/whisper-large-v3-finetuned-for-ATC"

# Initialize model on GPU with High Precision
# We use float16 for a balance of speed/memory, but you can use float32 for max precision if VRAM allows.
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Parameters optimized for ATC accuracy
segments, info = model.transcribe(
    "atc_audio_sample.wav",
    beam_size=10,        # Higher than default (5). Explores more paths to find the most accurate sequence.
    best_of=5,          # Generates multiple candidates and chooses the best. Default is usually lower.
    temperature=0,      # Force greedy decoding to prevent "creative" hallucinations in technical ATC terms.
    patience=2.0,       # Increased from default (1.0). Allows more time to explore deeper in beam search.
    condition_on_previous_text=False, # Vital for ATC: prevents errors from one transmission leaking into the next.
    language="en"       # Explicitly set to English to avoid auto-detection overhead or errors.
)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")