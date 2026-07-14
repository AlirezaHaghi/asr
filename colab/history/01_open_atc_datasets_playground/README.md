# 01 - Open ATC datasets playground

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

This is the messy dataset desk: ATCO2, the Jacktol combined set, and local copies of ATCOSIM or UWB-ATCC can all be poked with the same small tools. Nothing here pretends that train/test splits are interchangeable.

## حرف خودمونی

خلاصه این پوشه اینه: دیتاست رو می‌ریزیم وسط، ستون‌ها و splitها رو نگاه می‌کنیم و قبل از مدل‌بازی می‌فهمیم اصلاً چی دستمونه. اگه اسم ستون صدا یا متن فرق داشت، کد غر نمی‌زنه؛ گزارشش می‌کنه تا خودمون جمعش کنیم. عدد آماده و قلابی هم اینجا نداریم، باید اجرا کنی و خروجی همون اجرای خودت رو ببینی.

## چیزهایی که توی پوشه هست

- `dataset_catalog.json` - فهرست اولیه؛ شناسه‌های تأییدنشده عمداً خالی مانده‌اند.
- `list_dataset_cards.py` - اطلاعات card را موقع اجرا از Hugging Face می‌گیرد یا آفلاین فقط catalog را چاپ می‌کند.
- `load_any_atc_dataset.py` - یک loader سبک برای Hugging Face یا JSON/JSONL محلی.
- `compare_dataset_schemas.py` - مقایسه ستون، split، متن و duration چند منبع.
- `build_cross_dataset_manifest.py` - manifest مشترک بدون کپی کردن خود audio.
- سه notebook برای ور رفتن با schema، نمونه‌ها و cardها.

## چند اجرای ساده

```bash
python list_dataset_cards.py
python load_any_atc_dataset.py Jzuluaga/atco2_corpus_1h --split test --limit 5
python load_any_atc_dataset.py jacktol/ATC-ASR-Dataset --split test --limit 5
python compare_dataset_schemas.py --dataset Jzuluaga/atco2_corpus_1h::test --dataset jacktol/ATC-ASR-Dataset::test
python build_cross_dataset_manifest.py --dataset Jzuluaga/atco2_corpus_1h::test --limit 20 --output manifest.jsonl
```

ATCOSIM and UWB-ATCC are represented as local-source slots until an exact public dataset repository is supplied. Model-card WERs from different splits are not directly compared here.
