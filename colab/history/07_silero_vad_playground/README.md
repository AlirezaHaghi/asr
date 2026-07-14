# 07 - Silero and energy VAD playground

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

The old VAD sweep and separate VAD ablation were merged here. It now covers Silero thresholds, a plain energy baseline, segment export, ASR/no-ASR comparison, and paired prediction analysis.

## داستانش اینه

VAD بعضی وقتا سکوت رو جمع می‌کنه و کار مدل رو راحت می‌کنه، بعضی وقتا هم اول و آخر کلمه رو می‌جوه. روی ATCO2 که clipها از قبل کوتاهن، segment زیاد لزوماً شاهکار نیست. threshold رو آروم جابه‌جا کن، waveform رو ببین، بعد WER و deletion رو چک کن.

## چیزهایی که می‌شه باهاش ور رفت

- `inspect_vad.py` و `export_segments.py` برای Silero.
- `energy_vad.py` برای baseline خیلی ساده و قابل فهم.
- `compare_vad_asr.py` برای همان WAV با/بدون VAD.
- `compare_vad_runs.py` برای predictionهای جفت‌شده.
- دو پیاده‌سازی segment و چهار notebook برای threshold، waveform و ablation.

```bash
python inspect_vad.py sample.wav --thresholds 0.25 0.5 0.75 --output vad_sweep.json
python energy_vad.py sample.wav --output energy_segments.json
python export_segments.py sample.wav --output-dir vad_segments --threshold 0.5
python compare_vad_asr.py sample.wav --output vad_asr_comparison.json
python compare_vad_runs.py no_vad.json vad.json --output vad_comparison.json
```

All VAD paths must first convert audio to 16 kHz mono. A short-segment threshold is expressed in milliseconds/samples explicitly; the misleading old “25 seconds” comment is not reused.
