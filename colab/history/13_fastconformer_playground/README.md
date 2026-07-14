# 13 - NeMo FastConformer playground

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

This playground targets `niclaswue/youtube-atc-fastconformer`, a NeMo FastConformer hybrid RNNT-CTC checkpoint supplied in the review notes, plus a base NeMo reference for configuration comparison.

## به زبون راحت

این مدل برای speed و صدای رادیویی جذابه، ولی NeMo با Transformers یکی نیست؛ loader و decoder خودش رو می‌خواد. پس اینجا همه‌چی رو نچپوندیم توی یک `pipeline`. اول config رو نگاه می‌کنیم، بعد یه فایل، بعد یه folder. اگر checkpoint فرمت `.nemo` یا cache خاص خواست، خطاش صاف میاد بیرون.

## فایل‌ها

- `inspect_fastconformer.py` - config/card/cache probe بدون inference اجباری.
- `fastconformer_transcribe.py` - یک یا چند WAV با `ASRModel.from_pretrained`.
- `fastconformer_batch.py` - batch و JSONL failure log.
- `decoder_mode_plan.py` - طرح مقایسه RNNT و CTC بدون ادعای API ثابت.
- دو notebook برای NeMo setup و decoder inspection.

```bash
python inspect_fastconformer.py --model niclaswue/youtube-atc-fastconformer
python fastconformer_transcribe.py sample.wav --model niclaswue/youtube-atc-fastconformer
python fastconformer_batch.py wavs/ --model niclaswue/youtube-atc-fastconformer --output run.jsonl
python decoder_mode_plan.py
```

The card-reported number in the user notes must be checked against its exact ATCO2 setup before comparison with the delivered 871-sample benchmark.
