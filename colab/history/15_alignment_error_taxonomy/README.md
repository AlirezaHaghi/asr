# 15 - Alignment and error taxonomy / هم‌ترازی و دسته‌بندی خطا

Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

## چرا این‌قدر گیر می‌دیم؟

اشتباه یه کلمه معمولی با اشتباه callsign یا runway یه وزن عملیاتی نداره. alignment رو خرد می‌کنیم تا سوتی مهم مدل لای WER کلی قایم نشه.

## Purpose / هدف

This independent experiment reconstructs word-level alignment and reviews ATC error types without embedding reported results. Its backtrace represents insertions with a null reference token, preventing a single extra hypothesis word from shifting all subsequent pairs.

این پوشه خطاهای رونویسی را با هم‌ترازی درست بررسی می‌کند و نوع خطا را به number، command، callsign یا other تقسیم می‌کند. داده باید توسط کاربر ارائه شود.

## Files / فایل‌ها

- `align_words.py`: minimum-edit word alignment for one pair or a prediction JSON file.
- `classify_errors.py`: insertion-aware extraction, domain category, and transparent severity rules.
- `render_error_report.py`: Markdown summary from a generated taxonomy JSON file.
- `01_alignment_matrix.ipynb`: executable DP/backtrace demonstration.
- `02_taxonomy_explorer.ipynb`: explores a user-selected taxonomy JSON via `ERROR_JSON`.

## Commands / اجرا

```bash
python align_words.py --reference "contact tower" --hypothesis "contact the tower" --output alignment.json
python align_words.py --input predictions.json --output alignments.json
python classify_errors.py predictions.json --output error_taxonomy.json
python render_error_report.py error_taxonomy.json --output error_report.md
```

Prediction files accept a list or an object with `records`/`predictions`, using common reference and hypothesis field names.

## Expected outputs / خروجی مورد انتظار

Alignment JSON contains `equal`, `substitution`, `deletion`, and `insertion` operations. Taxonomy JSON includes source record IDs, aligned tokens, category, severity, and aggregate counts calculated at run time. The report only summarizes that generated file.

خروجی‌ها شامل جفت کلمات و دلیل دسته‌بندی هستند؛ هیچ جدول نتیجه از قبل ذخیره نشده است.

## Limitations / محدودیت‌ها

The taxonomy is rule-based, not an aviation safety assessment. Callsigns are difficult to infer without metadata. Multiple equally minimal alignments can produce different local labels. Review audio and operational context before treating severity as meaningful.
