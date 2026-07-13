# 10 - Voxtral ATC playground

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

This folder explores a heavier audio-language-model route instead of conventional Whisper/CTC decoding. The main supplied checkpoint is `pphilip/voxtral-3B-atc-transcribe`; the base entry is kept beside it for adapter/config inspection.

## قصه چیه؟

Voxtral یه مدل سبکِ لپ‌تاپی نیست؛ سه میلیارد پارامتر یعنی باید قبل از دانلود یه حساب سرانگشتی VRAM بزنیم. اینجا اول prompt و config رو می‌چینیم، بعد اگه GPU جواب داد transcription واقعی می‌گیریم. اگه مدل card یا API عوض شده بود، اسکریپت inspect همون موقع لو می‌ده؛ الکی وانمود نمی‌کنیم اجرا شده.

## فایل‌های این مدل‌بازی

- `voxtral_catalog.json` - base/fine-tuned IDs و یادداشت‌های غیرقابل‌مقایسه.
- `inspect_voxtral_card.py` - config و metadata زنده یا local cache.
- `build_voxtral_prompts.py` - promptهای کوتاه ATC، آزاد و JSON conversation.
- `estimate_voxtral_memory.py` - تخمین خیلی تقریبی وزن/VRAM.
- `voxtral_transcribe.py` - loader محافظه‌کار برای کلاس‌های جدید Transformers.
- سه notebook هشت-cell برای prompt، memory و یک WAV.

```bash
python inspect_voxtral_card.py --model pphilip/voxtral-3B-atc-transcribe
python build_voxtral_prompts.py --style strict
python estimate_voxtral_memory.py --parameters-b 3 --dtype bf16
python voxtral_transcribe.py sample.wav --model pphilip/voxtral-3B-atc-transcribe
```

Card-reported WER supplied with the request is not inserted as an executed result. Re-evaluate on the same untouched clips used for every other model.
