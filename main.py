import sys
import os
import json
from pathlib import Path
from tqdm import tqdm
from asr_engine import ASREngine
from vad_utils import VADProcessor
from text_normalization import normalize_atc_text

def process_pipeline(src_dir_name: str):
    src_path = Path(src_dir_name)
    if not src_path.is_dir():
        print(f"Error: {src_path} is not a valid directory.")
        sys.exit(1)
        
    dest_path = Path(src_dir_name + "-transcripts")
    dest_path.mkdir(exist_ok=True)
    logs_path = dest_path / "execution_logs.json"

    audio_files = [os.path.join(src_path, f) for f in os.listdir(src_path) if f.lower().endswith(".wav")]
    
    # Initialize Pipeline Components
    print("Loading VAD and ASR Models...")
    vad = VADProcessor()
    asr = ASREngine(model_name="large-v3") # یا "fjmgAI/whisper-large-v3-ATC"
    execution_logs = []

    with tqdm(total=len(audio_files), desc="Running End-to-End ATC Pipeline") as pbar:
        for file in audio_files:
            file_name = os.path.splitext(os.path.basename(file))[0]
            try:
                # 1. VAD: استخراج بخش‌های دارای صحبت
                wav, timestamps = vad.get_speech_segments(file)
                full_transcript = []
                
                # 2. ASR: تبدیل هر بخش به متن
                for ts in timestamps:
                    start_sample = int(ts["start"] * 16000)
                    end_sample = int(ts["end"] * 16000)
                    segment = wav[start_sample:end_sample].numpy()
                    raw_text = asr.transcribe_segment(segment)
                    
                    # 3. Normalization: استانداردسازی متن
                    normalized_text = normalize_atc_text(raw_text)
                    full_transcript.append(normalized_text)
                
                final_text = " ".join(full_transcript)
                
                # ذخیره خروجی
                with open(dest_path / f"{file_name}.txt", "w", encoding="utf-8") as f:
                    f.write(final_text)
                    
                execution_logs.append({"file": file_name, "status": "success", "segments": len(timestamps)})
                
            except Exception as e:
                execution_logs.append({"file": file_name, "status": "failed", "error": str(e)})
            
            pbar.update(1)
            
    # ذخیره لاگ‌های اجرایی (Artifacts)
    with open(logs_path, "w") as f:
        json.dump(execution_logs, f, indent=4)
    print(f"Processing complete. Logs saved to {logs_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <source_audio_dir>")
        sys.exit(1)
    process_pipeline(sys.argv[1])