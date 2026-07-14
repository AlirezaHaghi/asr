# 12 - Cross-family ATC benchmark bench

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

This directory is intentionally bigger and messier than the others. Environment snapshots and cache/fingerprint tools were merged here because they matter most when Whisper, CTC, NeMo and audio-language models are compared in one run.

## رفیق، این یکی میز جمع‌بندیه

هر مدل یه ساز می‌زنه: Transformers pipeline، `AutoModelForCTC`، NeMo، یا audio-LLM. اینجا قرار نیست همه رو با یه loader زورکی اجرا کنیم. کار این پوشه اینه که plan و schema و metric رو یکی کنه تا آخرش سیب رو با پرتقال مقایسه نکنیم. هر ردیف باید dataset، split، normalizer، زمان و خطاش معلوم باشه.

## خانواده‌ها

- Whisper ATC fine-tunes
- Wav2Vec2/XLS-R CTC
- `niclaswue/youtube-atc-fastconformer`
- `qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC`
- `pphilip/voxtral-3B-atc-transcribe`
- `suideepmax/canary-qwen-2.5b-atc-lora`

## ابزارها

- `cross_family_catalog.json` و `make_benchmark_matrix.py` برای plan.
- `unified_result_schema.py` برای validate/convert کردن JSONL.
- `compare_family_results.py` برای corpus WER/CER/RTF روی خروجی‌های واقعی.
- فایل‌های cache, fingerprint, dependency و GPU که از پوشه‌های ساده قبلی اینجا merge شدند.
- notebookهای معماری، environment، cache و benchmark scratchpad.

```bash
python make_benchmark_matrix.py --dataset Jzuluaga/atco2_corpus_1h --split test --output benchmark_plan.json
python unified_result_schema.py validate predictions.jsonl
python compare_family_results.py whisper.jsonl ctc.jsonl nemo.jsonl
python fingerprint_config.py benchmark_plan.json
```

Model-card WERs live in notes only; ranking requires rerunning every model on the same untouched manifest.
