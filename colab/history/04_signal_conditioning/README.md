# 04 — Signal conditioning / آماده‌سازی سیگنال

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

## رک و راست

noise gate ممکنه توی هدفون صدا رو باحال‌تر کنه ولی s و f رو هم بجوه. پس «به گوشم بهتر شد» نتیجه ASR نیست؛ قبل و بعدش باید دوباره مدل رو زد.

This isolated study tests conservative preprocessing candidates before ASR: peak normalization, pre-emphasis, and a frame-energy noise gate. هدف، بررسی اثر تبدیل سیگنال است؛ این مراحل در نتایج نهایی به عنوان بهبود قطعی ادعا نمی‌شوند.

## Files / فایل‌ها

- `peak_normalize.py`: gain-only normalization to a requested dBFS peak with clipping protection.
- `preemphasis_filter.py`: first-order pre-emphasis and optional inverse reconstruction check.
- `adaptive_noise_gate.py`: frame RMS noise-floor estimate, soft gating, and smoothed gain envelope.
- `conditioning_sandbox.ipynb`: synthetic radio-like mixture and measurable conditioning diagnostics.
- `snr_threshold_sweep.ipynb`: repeatable gate-threshold sensitivity study on synthetic speech-like audio.

## Run / اجرا

```bash
python peak_normalize.py input.wav normalized.wav --target-dbfs -3
python preemphasis_filter.py input.wav preemphasized.wav --coefficient 0.97
python adaptive_noise_gate.py input.wav gated.wav --noise-percentile 20 --margin-db 6
```

Each command writes a new WAV and prints measured parameters from that rerun. خروجی‌ها باید دوباره ساخته شوند و عدد از پیش ثبت‌شده‌ای وجود ندارد.

## Limitations / محدودیت‌ها

Conditioning can remove quiet consonants, amplify noise, or shift a model away from its training distribution. Synthetic waveforms are diagnostic only; WER must be measured separately on real labeled audio. تنظیمات این بخش production-ready نیستند.
