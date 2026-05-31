import nemo.collections.asr as nemo_asr

# Load the fine-tuned Parakeet-TDT model
model_name = "qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC"
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)

# Transcription with accuracy-focused config
# Parakeet-TDT handles internal parameters differently than Whisper
files = ["atc_audio_sample.wav"]

# We adjust the decoding strategy for the Transducer architecture
# Note: TDT models are inherently fast; we focus on 'greedy' vs 'beam'
transcriptions = asr_model.transcribe(
    paths2audio_files=files,
    batch_size=4,
    return_hypotheses=False # Set to True if you want to manually inspect alternative scores
)

print(f"Transcription: {transcriptions[0]}")