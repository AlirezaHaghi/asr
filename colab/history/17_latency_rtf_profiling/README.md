# 17 - Latency and RTF profiling / سنجش زمان و ضریب بلادرنگ

Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

## سرعت رو قاطی نکن

load مدل، warm-up و inference سه تا چیز جدا هستن. RTF و throughput رو هم با batch و device خودشون ثبت می‌کنیم تا عدد قشنگ ولی بی‌مصرف درنیاد.

## Purpose / هدف

This independent experiment measures model load time, per-file wall latency, audio duration, and real-time factor (RTF = wall seconds / audio seconds) on user-supplied audio. Every record carries the current invocation's model, device, and beam provenance plus a canonical configuration fingerprint.

هدف این پوشه اندازه‌گیری سرعت واقعی روی سیستم کاربر است. خطاهای هر فایل داخل JSON ثبت می‌شوند و با نتیجه موفق اشتباه گرفته نمی‌شوند.

## Files / فایل‌ها

- `profile_transcriber.py`: profiles a Hugging Face ASR pipeline over local audio and logs per-file exceptions.
- `summarize_profile.py`: computes descriptive latency/RTF statistics from one generated profile.
- `compare_profiles.py`: compares multiple profile JSON files without mixing their configuration provenance.
- `01_rtf_measurement_lab.ipynb`: executable wall-time/RTF method demonstration.
- `02_profile_explorer.ipynb`: explores a user-selected profile via `PROFILE_JSON`.

## Commands / اجرا

```bash
python -m pip install -r requirements.txt
python profile_transcriber.py ./wav --model MODEL_ID --beam-size 1 --device -1 --output profile_beam1.json
python summarize_profile.py profile_beam1.json --output profile_summary.json
python compare_profiles.py "profile_*.json" --output profile_comparison.json
```

`--device -1` means CPU; nonnegative values select a CUDA device. Use a new output file for every current invocation.

## Expected outputs / خروجی مورد انتظار

The raw profile includes current-run config/fingerprint, load time, successful measurements, and structured failures (`exception_type`, message, traceback). Summaries report counts, mean/median/p95 wall time and RTF from successful positive-duration files only.

خروجی‌ها در زمان اجرا ساخته می‌شوند و هیچ سرعت یا سخت‌افزار فرضی در کد ثبت نشده است.

## Limitations / محدودیت‌ها

Latency depends on hardware, model cache, library versions, file duration, batch size, and warm-up. CUDA synchronization reduces asynchronous timing error but does not make machines comparable. File I/O and model loading are reported separately. This script profiles inference behavior; it is not a production load test.
