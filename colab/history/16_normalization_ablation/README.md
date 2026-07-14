# 16 - ATC normalization ablation / آزمون قواعد نرمال‌سازی

Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

## بازی با rule، ولی تمیز

یه rule رو خاموش می‌کنیم و دوباره metric می‌گیریم. اگه همزمان ده تا regex عوض شه، دیگه معلوم نیست کدومش اثر گذاشته.

## Purpose / هدف

This independent reconstruction separates scoring normalization from display formatting and measures rule-family effects on supplied references/hypotheses. It covers ICAO variants, flight levels, runways, digit sequences, frequencies, punctuation, and selected airline spacing variants.

هدف این بخش این است که نشان دهد کدام قانون نرمال‌سازی روی WER اثر دارد. متن اصلی حفظ می‌شود و نتایج فقط از فایل ورودی کاربر محاسبه می‌شوند.

## Files / فایل‌ها

- `normalize_for_wer.py`: scoring-oriented normalization for text, line files, or JSON arrays.
- `normalize_for_display.py`: compresses spoken flight levels, runways, and decimal frequencies for readable output.
- `normalization_ablation.py`: compares `raw`, `punctuation`, `icao`, and `atc` profiles on supplied predictions.
- `01_rule_walkthrough.ipynb`: executable examples labeled as rule demonstrations, not benchmark findings.
- `02_corpus_ablation.ipynb`: inventories and measures a user-selected prediction file via `PREDICTIONS_JSON`.

## Commands / اجرا

```bash
python normalize_for_wer.py --text "Descend FL180, runway 28L" --output wer_text.json
python normalize_for_display.py --text "contact one two seven decimal four" --output display_text.json
python normalization_ablation.py predictions.json --output normalization_ablation.json
```

For line mode use `--input text.txt`; for JSON mode add `--json-field hypothesis_raw`. Prediction records accept the common reference/hypothesis field names documented in the scripts.

## Expected outputs / خروجی مورد انتظار

Normalizer outputs preserve before/after pairs. The ablation output gives reference-word counts, edit counts, and WER for each profile, computed fresh from user data.

خروجی مورد انتظار جدول پروفایل‌ها است؛ هیچ پروفایل به عنوان برنده از قبل اعلام نشده است.

## Limitations / محدودیت‌ها

Regex rules cannot understand all ATC context, accents, multilingual speech, callsigns, or ambiguous number groupings. Display normalization is for readability and must not silently replace the scoring representation. Always retain raw text for audit.
