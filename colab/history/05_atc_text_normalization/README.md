# 05 — ATC text normalization / نرمال‌سازی متن هوانوردی

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

## خودمونی‌ش

Normalizer قرار نیست transcript رو حدس بزنه؛ فقط شکل‌های هم‌معنی رو یکی می‌کنه. فایل‌های display هم اینجا merge شدن تا FL180 و RWY28L و 127.4 کنار ruleهای WER باشن.

This bundle reconstructs and tests the final WER normalizer: case/punctuation cleanup, ICAO pronunciations, airline token joins, FL/RWY abbreviations, and digit expansion. هدف، جدا کردن اثر نرمال‌سازی متن از کیفیت واقعی مدل است.

## Files / فایل‌ها

- `normalize_transcript.py`: CLI normalizer for one string or a UTF-8 line corpus.
- `normalization_ablation.py`: recomputes WER after progressively enabling normalization rule families.
- `corpus_normalization_audit.py`: counts changed lines and rule-trigger patterns without ASR inference.
- `normalizer_rulebook.ipynb`: executable rule examples and idempotence check.
- `wer_normalization_study.ipynb`: small labeled raw-versus-normalized WER demonstration.

## Run / اجرا

```bash
python normalize_transcript.py --text "Descend FL180, runway 28L"
python normalize_transcript.py --input transcripts.txt --output normalized.txt
python normalization_ablation.py --jsonl references_and_hypotheses.jsonl --output ablation.json
python corpus_normalization_audit.py transcripts.txt --output audit.json
```

For JSONL ablation, each row needs `reference` and `hypothesis`. All metrics are computed on rerun; none are asserted in this archive. آمار فقط پس از اجرای کد تولید می‌شود.

## Limitations / محدودیت‌ها

Normalization can hide semantically important formatting errors, and per-digit expansion is not a complete spoken-number grammar. Rules are English/ATC-specific and may need revision for new airlines or languages. این مرحله جایگزین ارزیابی انسانی نیست.
