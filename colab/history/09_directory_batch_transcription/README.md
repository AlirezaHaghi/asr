# 09 - Recursive WAV batch transcription / رونویسی پوشه‌ای

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

## توی عمل

یه folder واقعی همیشه یه فایل خراب یا sample rate ناجور داره. batch خوب خطا رو می‌نویسه، ادامه می‌ده و دفعه بعد از اول شروع نمی‌کنه؛ با یه glob خوشگل کار تموم نیست.

This standalone folder recreates a directory-oriented research workflow: inventory WAV files, transcribe them with restartable JSONL output, and summarize only observed records. قابلیت پردازش تمام WAVهای یک پوشه یک exploration بازسازی‌شده است و ادعا نمی‌شود که عیناً در نوت‌بوک نهایی وجود داشته است.

## Files / فایل‌ها

- `build_manifest.py`: recursively inventory WAV files with hashes and optional sidecar references.
- `batch_transcribe.py`: deterministic-order, resumable Whisper inference with explicit failure records.
- `summarize_run.py`: aggregate status, duration, real-time factor, and optional corpus WER/CER/S-D-I.
- `manifest_audit.ipynb`: inspect distributions and duplicate hashes.
- `batch_results_analysis.ipynb`: analyze newly generated JSONL results.

## Commands / اجرا

```bash
python -m pip install -r requirements.txt
python build_manifest.py ./wav_directory --transcripts-dir ./transcripts --output manifest.jsonl
python batch_transcribe.py manifest.jsonl --output predictions.jsonl --beams 5
python summarize_run.py predictions.jsonl --output summary.json
```

Expected outputs are a hash-bearing input manifest, one success/error record per WAV, and an aggregate summary. Resume mode skips only IDs already recorded as successful. خروجی واقعی باید دوباره تولید شود.

## Limitations / محدودیت‌ها

Reference matching assumes parallel relative paths ending in `.txt`; projects with other schemas need adaptation. Recursive scanning does not imply that all files form a valid test set. A model download and sufficient GPU/RAM are required. The scripts do not manufacture missing transcripts or performance numbers.
