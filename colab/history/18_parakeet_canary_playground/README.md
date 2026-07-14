# 18 - Parakeet TDT and Canary-Qwen playground

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

The old cache-only folder is gone. This one is about NVIDIA speech models: the ATC-fine-tuned Parakeet checkpoint, its base model, and the supplied Canary-Qwen LoRA adaptation.

## خودمونی بخون

Parakeet برای throughput خیلی وسوسه‌کننده‌ست و TDT با Whisper فرق جدی داره. Canary-Qwen هم speech-LLMـه و prompt و adapter داستان خودش رو داره. اینجا اول card/config رو باز می‌کنیم، بعد Parakeet رو روی WAV می‌زنیم، و برای Canary تا وقتی API دقیق کارت تأیید نشده الکی loader جادویی نمی‌سازیم؛ adapter و prompt رو شفاف بررسی می‌کنیم.

## فایل‌ها

- `nemo_model_catalog.json` - base/fine-tuned IDs و caveatهای train/eval.
- `inspect_nemo_cards.py` - metadata و فایل‌های checkpoint.
- `parakeet_transcribe.py` - single/batch transcription با NeMo.
- `compare_parakeet_base_finetuned.py` - همان audio برای base و fine-tuned.
- `canary_lora_probe.py` - adapter/config/prompt inspection، نه نتیجه‌سازی.
- سه notebook: Parakeet quick lab، base-vs-fine-tune و Canary adapter scratchpad.

```bash
python inspect_nemo_cards.py
python parakeet_transcribe.py sample.wav --model qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC
python compare_parakeet_base_finetuned.py sample.wav
python canary_lora_probe.py --model suideepmax/canary-qwen-2.5b-atc-lora
```

The supplied 5.99% and 23.32% notes belong to their own dataset/split definitions. They are not direct replacements for an untouched ATCO2-1h result.
