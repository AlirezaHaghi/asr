# ATC-ASR Benchmark — Google Colab Notebook
# ─────────────────────────────────────────────────────────────
# Runtime → Change runtime type → T4 GPU
# اجرا: هر cell را به ترتیب اجرا کن
# ─────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════
# CELL 1 ─ Mount Drive + Clone Repo
# ══════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount("/content/drive")

DRIVE_DIR = "/content/drive/MyDrive/atc_asr_results"
REPO_URL  = "https://github.com/AlirezaHaghi/asr.git"  # ← repo خودت

import os, subprocess
os.makedirs(DRIVE_DIR, exist_ok=True)

# Clone یا pull
if not os.path.exists("/content/atc-asr"):
    subprocess.run(["git", "clone", REPO_URL, "/content/atc-asr"], check=True)
else:
    subprocess.run(["git", "-C", "/content/atc-asr", "pull"], check=True)

os.chdir("/content/atc-asr")
print("✓ Repo آماده است")


# ══════════════════════════════════════════════════════════════
# CELL 2 ─ نصب وابستگی‌ها
# ══════════════════════════════════════════════════════════════
# %%capture
import subprocess
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ══════════════════════════════════════════════════════════════
# CELL 3 ─ اجرای benchmark
# ══════════════════════════════════════════════════════════════
# این cell تمام کار را انجام می‌دهد:
# دانلود ATCO2-1h → inference دو مدل → WER/CER → error analysis
import subprocess
result = subprocess.run(["python", "evaluate.py"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)


# ══════════════════════════════════════════════════════════════
# CELL 4 ─ نمایش نتایج
# ══════════════════════════════════════════════════════════════
import json, pandas as pd

with open("eval_output/benchmark_results.json") as f:
    results = json.load(f)

df = pd.DataFrame([{
    "مدل": r["run_name"],
    "WER": f"{r['WER']:.2%}",
    "CER": f"{r['CER']:.2%}",
    "S": r["S"], "D": r["D"], "I": r["I"],
    "نمونه": r["samples"],
} for r in results])
print(df.to_string(index=False))

print("\n─── Error Analysis ───")
with open("eval_output/error_analysis.txt", encoding="utf-8") as f:
    print(f.read()[:3000])  # ۳۰۰۰ کاراکتر اول


# ══════════════════════════════════════════════════════════════
# CELL 5 ─ کپی نتایج به Google Drive
# ══════════════════════════════════════════════════════════════
import shutil, datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
dest = f"{DRIVE_DIR}/run_{timestamp}"
shutil.copytree("eval_output", dest)
print(f"✓ نتایج ذخیره شد در Drive:")
print(f"  {dest}/")
for f in os.listdir(dest):
    size = os.path.getsize(f"{dest}/{f}")
    print(f"  {f}  ({size:,} bytes)")
