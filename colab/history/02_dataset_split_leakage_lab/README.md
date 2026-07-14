# 02 - Dataset split and leakage lab

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

Folder 01 opens different corpora; this folder is stricter. It freezes manifests and checks whether train/eval rows share audio hashes, IDs, or normalized transcripts.

## خودمونی

WER دو درصد خیلی باحاله، ولی اگه همون صدا یا همون جمله توی train و test چرخیده باشه به درد نمی‌خوره. اینجا قبل از ذوق‌زدگی hash و ID و متن رو می‌کوبیم به هم تا overlap معلوم شه. هشدار می‌ده؛ خودش الکی حکم leakage قطعی صادر نمی‌کنه.

## ابزارها

- `inspect_dataset_schema.py` - schema و چند نمونه ATCO2.
- `build_sample_manifest.py` - manifest ثابت با ID و metadata.
- `dataset_quality_checks.py` - missing text/audio، rate و duration.
- `detect_manifest_overlap.py` - ID/audio/text overlap بین دو JSONL.
- notebookهای schema، duration و overlap.

```bash
python inspect_dataset_schema.py --split test --limit 5 --output schema_report.json
python build_sample_manifest.py --split test --limit 100 --output eval_manifest.jsonl
python dataset_quality_checks.py --split test --limit 871 --output quality_checks.json
python detect_manifest_overlap.py train_manifest.jsonl eval_manifest.jsonl --output overlap_report.json
```

Text overlap alone can be normal in phraseology; audio hashes and source recording IDs are stronger leakage signals.
