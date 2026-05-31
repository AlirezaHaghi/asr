from transformers.pipelines import pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Using custom `forced_decoder_ids`")

pipe = pipeline(
    "automatic-speech-recognition",
    model="jacktol/whisper-medium.en-fine-tuned-for-ATC",
    device=0,
    chunk_length_s=30,
    stride_length_s=5,
)

def transcribe(src_path: str):
    result = pipe(src_path)
    return result["text"]
        
from transformers import pipeline
import torch

class ASREngine:
    def __init__(self, model_id="fjmgAI/whisper-large-v3-ATC"):
        print(f"Loading ASR Model: {model_id}")
        
        # استفاده مجدد از pipeline قدرتمند Hugging Face
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,
            chunk_length_s=30,
            stride_length_s=5,
        )
        
        # پارامترهای بهینه برای محیط ATC (اعمال شده روی Transformers)
        self.generate_kwargs = {
            "language": "en",
            "task": "transcribe",
            "temperature": 0.0,          # قطعی‌ترین پیش‌بینی (پیشنهاد مستندات)
            "num_beams": 5,              # دقت بالاتر در جستجو
            "condition_on_prev_tokens": False # جلوگیری از انتشار خطا بین چانک‌ها
        }

    def transcribe_segment(self, audio_input):
        """
        دریافت مسیر فایل صوتی یا آرایه numpy و برگرداندن متن
        """
        # پاس دادن تنظیمات به عنوان generate_kwargs
        result = self.pipe(audio_input, generate_kwargs=self.generate_kwargs)
        return result["text"]
