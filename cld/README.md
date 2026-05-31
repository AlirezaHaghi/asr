# ATC-ASR Pipeline

سیستم تشخیص گفتار خودکار (ASR) برای حوزه کنترل ترافیک هوایی (ATC)

مدل استفاده‌شده: [`jacktol/whisper-medium.en-fine-tuned-for-ATC`](https://huggingface.co/jacktol/whisper-medium.en-fine-tuned-for-ATC)

---

## ساختار پروژه

```
├── asr_engine.py        موتور ASR (inference با transformers pipeline)
├── main.py              پایپ‌لاین تولید — پوشه صوتی → پوشه transcript
├── evaluate.py          benchmark روی ATCO2-test-set-1h
├── text_normalizer.py   ATC text normalization
├── requirements.txt
└── ATC_Benchmark_Colab.py   دفترچه Google Colab
```

---

## نصب

```bash
pip install -r requirements.txt
```

---

## استفاده

### ۱. تبدیل صوت به متن (production)

```bash
python main.py /path/to/audio_folder
```

خروجی در پوشه `audio_folder-transcripts/` ذخیره می‌شود.

### ۲. اجرای benchmark روی ATCO2-1h

```bash
python evaluate.py
```

خروجی‌ها در `eval_output/`:

| فایل | محتوا |
|------|-------|
| `benchmark_results.json` | WER/CER + S/D/I دو مدل/config |
| `predictions_run1.json` | خروجی کامل مدل ۱ |
| `predictions_run2.json` | خروجی کامل مدل ۲ |
| `error_analysis.txt` | ۳۰+ نمونه خطای دسته‌بندی‌شده |

### ۳. اجرا در Google Colab

1. فایل `ATC_Benchmark_Colab.py` را در Colab باز کن
2. آدرس repo را در `REPO_URL` وارد کن
3. Runtime → **T4 GPU**
4. هر cell را به ترتیب اجرا کن

---

## نیازمندی‌های سخت‌افزاری

| محیط | حداقل |
|------|-------|
| Inference (medium) | 8 GB VRAM یا CPU (کند) |
| Google Colab Free | T4 — کافی است |
| RAM سیستم | 8 GB |

---

## دیتاست ارزیابی

[ATCO2-test-set-1h](https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h) — رایگان، معتبرترین benchmark در حوزه ATC-ASR

---

## بازتولید نتایج

```bash
# بررسی نسخه‌ها
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
pip freeze > requirements_exact.txt
```
