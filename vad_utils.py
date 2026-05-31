import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

class VADProcessor:
    def __init__(self):
        self.model = load_silero_vad()

    def get_speech_segments(self, audio_path: str):
        """
        دریافت زمان‌های شروع و پایان بخش‌های دارای گفتار در یک فایل صوتی
        """
        wav = read_audio(audio_path, sampling_rate=16000)
        timestamps = get_speech_timestamps(
            wav, 
            self.model,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            return_seconds=True
        )
        return wav, timestamps