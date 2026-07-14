# 06 - Hotword, callsign and number playground

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

This folder turns the final report's callsign/number recommendation into small experiments: lexicon cleanup, Whisper prompt construction, CTC decoder vocabulary bias, and nasty numeric minimal pairs.

## خیلی ساده بگم

مدل ممکنه صدای «six zero» رو خوب بشنوه ولی آخرش یه callsign عجیب تحویل بده. اینجا می‌خوایم یه کم هلش بدیم سمت کلمه‌های درست، نه اینکه جواب رو زورکی توی دهن مدل بذاریم. هر چی چاپ می‌شه مال اجرای خودته؛ این فایل‌ها نتیجه از پیش پخته ندارن.

## فایل‌ها همین دور و برن

- `callsign_lexicon.py` - تمیز کردن airline/callsign و ساخت unigram list.
- `whisper_hotword_prompt.py` - prompt کوتاه برای Whisper و اجرای اختیاری روی WAV.
- `numeric_confusion_cases.py` - ساخت جمله‌های شبیه هم برای heading، FL، runway و frequency.
- `ctc_hotword_decoder.py` - decode کردن logits ذخیره‌شده با unigram/LM اختیاری.
- notebookها: یکی برای callsign، یکی برای عددها، یکی برای prompt.

```bash
python callsign_lexicon.py --input airlines.txt --output callsigns.json
python numeric_confusion_cases.py --output numeric_cases.jsonl
python whisper_hotword_prompt.py sample.wav --dry-run
python ctc_hotword_decoder.py logits.npy vocab.json --unigrams callsigns.txt
```

Prompt یا hotword ممکن است hallucination را بیشتر کند. مقایسه باید روی همان audio و همان normalizer انجام شود؛ صرفاً قشنگ‌تر شدن یک transcript معیار نیست.
