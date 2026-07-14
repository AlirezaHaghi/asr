# 03 — Audio I/O and resampling / ورودی‌خروجی و بازنمونه‌برداری صدا

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

## یه نکته خودمونی

فایل WAV هر چی اسمش باشه دلیل نمی‌شه واقعاً 16k mono باشه. اول header و channel و dtype رو می‌بینیم، بعد resample می‌کنیم؛ حدس زدن اینجا ممنوع.

This experiment isolates the assumptions hidden behind the final 16 kHz ASR/VAD path: WAV discovery, channel layout, subtype, amplitude, and deterministic resampling. هدف این است که ورودی صوتی قبل از مدل قابل بررسی و تکرار باشد.

## Files / فایل‌ها

- `audio_inventory.py`: recursively records WAV headers and lightweight signal statistics.
- `resample_audio.py`: mono/stereo-preserving polyphase resampling with explicit output subtype.
- `channel_dtype_probe.py`: per-channel peak, RMS, DC offset, clipping, and crest-factor analysis.
- `wav_io_roundtrip.ipynb`: generates a test tone and verifies a SoundFile write/read round trip.
- `resampling_study.ipynb`: compares 48 kHz→16 kHz polyphase resampling on a synthetic multitone.

## Run / اجرا

```bash
python audio_inventory.py ./wav_data --output audio_inventory.json
python channel_dtype_probe.py sample.wav --output channel_probe.json
python resample_audio.py sample.wav sample_16k.wav --target-rate 16000 --mono
```

Expected outputs are a JSON inventory/probe and newly encoded WAV, produced only after running. هیچ عدد یا فایل صوتی از قبل به عنوان نتیجه ادعا نشده است.

## Limitations / محدودیت‌ها

Synthetic tones do not represent radio noise, accents, or codec damage. Downmixing can cancel opposite-phase channels, and resampling cannot restore missing bandwidth. این ابزارها کیفیت رونویسی را تضمین نمی‌کنند.
