# 08 - Whisper ATC playground

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

This is no longer a one-file demo. It is a loose playground for base and ATC-fine-tuned Whisper checkpoints, decoding knobs, chunking, timestamps, prompt experiments, and same-audio checkpoint comparisons.

## خودمونی‌ش

اینجا قراره با Whisper ور بریم: یه بار greedy، یه بار beam، یه بار مدل کوچیک، یه بار مدل ATCOSIM. فقط حواسمون هست عددهای model card رو قاطی نکنیم؛ دیتاست‌ها یکی نیستن و WER پایین روی صدای تمیز لزوماً روی بی‌سیم واقعی جواب نمی‌ده.

## مدل‌هایی که توی catalog گذاشتیم

- `jacktol/whisper-large-v3-finetuned-for-ATC`
- `tclin/whisper-large-v3-turbo-atcosim`
- `youngsangroh/whisper-small-atco2-atcosim`
- `youngsangroh/whisper-large-atco2-atcosim`
- `jlvdoorn/whisper-medium-atco2`
- `fjmgAI/whisper-large-v3-ATC`

اعداد WER داخل catalog فقط یادداشت‌های داده‌شده توسط کاربرند و باید با card زنده و split واقعی دوباره چک شوند.

## شروع سریع

```bash
python inspect_whisper_cards.py
python transcribe_single.py sample.wav --model jacktol/whisper-large-v3-finetuned-for-ATC
python compare_whisper_checkpoints.py sample.wav --models jacktol/whisper-large-v3-finetuned-for-ATC tclin/whisper-large-v3-turbo-atcosim
python build_decoding_grid.py --beam-sizes 1 5 10
python run_decoding_ablation.py sample.wav --config decoding_grid.json
```

Notebookها عمداً شل‌وول و آزمایشگاهی‌اند؛ هر کدام دست‌کم هشت cell دارند و خروجی ذخیره‌شده ندارند.
