# 14 - WER/CER and S-D-I metrics / محاسبه معیارهای خطا

Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

## حواست به تله باشه

میانگین WER جمله‌ها با corpus WER یکی نیست. اینجا S و D و I رو جمع می‌کنیم، insertion اول جمله رو هم گم نمی‌کنیم و بعد عدد نهایی رو می‌سازیم.

## Purpose / هدف

This independent bundle recomputes corpus WER, CER, substitutions, deletions, insertions, and exact-match counts from user-supplied predictions. The dynamic-programming alignment explicitly keeps insertion operations instead of shifting later words.

هدف این بخش بررسی دوباره معیارها از روی فایل پیش‌بینی واقعی کاربر است. هیچ نتیجه نهایی پروژه داخل کد ثابت نشده و همه اعداد باید دوباره محاسبه شوند.

## Files / فایل‌ها

- `compute_sdi.py`: validates prediction records and writes corpus and per-record word/character metrics.
- `aggregate_runs.py`: compares multiple run files and reads beam provenance from the matching current-run config, with a SHA-256 config fingerprint.
- `validate_alignment_metrics.py`: deterministic alignment edge-case checks, including a middle insertion.
- `01_sdi_walkthrough.ipynb`: executable insertion-aware DP walkthrough.
- `02_run_metric_table.ipynb`: inspects user-supplied prediction files selected by `PREDICTION_GLOB`.

## Commands / اجرا

```bash
python -m pip install -r requirements.txt
python compute_sdi.py predictions.json --output metrics.json
python aggregate_runs.py "../inputs/predictions_*.json" --config-dir ../inputs/configs --output run_table.json
python validate_alignment_metrics.py --output metric_validation.json
```

Accepted text fields are `reference`/`reference_raw`/`ref` and `hypothesis`/`hypothesis_raw`/`hyp`. Files can be lists or objects containing `records` or `predictions`.

## Expected outputs / خروجی مورد انتظار

Outputs contain counts and ratios calculated from inputs, source paths, run IDs, matching beam settings, and config fingerprints. `aggregate_runs.py` never carries a previous run's beam value into the next row.

خروجی شامل WER و CER و تعداد S/D/I است و فقط از ورودی محاسبه می‌شود.

## Limitations / محدودیت‌ها

Metrics depend on the normalization already present in the input text. Empty-reference records have undefined WER and are rejected for corpus reporting. Tie-breaking chooses one valid minimum edit path, so S/D/I decomposition can differ between equally optimal paths while total edit distance remains equal.
