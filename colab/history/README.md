# ATC-ASR retrospective playground archive

Updated: **2026-07-13**

> Retrospective reconstruction created from the final notebook/report. It is not claimed to be an original timestamped development artifact; generated metrics require rerunning.

This tree is now organized as a mixed research playground rather than eighteen copies of the same folder template. The Python files and notebooks are the artifacts themselves, so the empty `artifacts/` directories were removed. Scripts accept explicit output filenames when an experiment is actually run.

## حرف خودمونی

این پوشه قراره شبیه میز کار واقعی باشه: یه جا سه تا notebook داریم، یه جا کلی script، یه جا catalog مدل، یه جا فقط metric و error analysis. همه‌چی اتوکشیده و کپی‌پیست نیست. با این حال تاریخ‌سازی هم نمی‌کنیم؛ هر جا نتیجه واقعی لازم است باید کد را اجرا کرد و log همان اجرا را نگه داشت.

## Playground map

| Directory | What gets tested |
|---|---|
| `01_open_atc_datasets_playground` | ATCO2, Jacktol combined data, local ATCOSIM/UWB manifests, schemas and cards |
| `02_dataset_split_leakage_lab` | frozen manifests, schema quality and train/eval overlap |
| `03_audio_io_resampling` | WAV metadata, channels, dtype and 8/16 kHz resampling |
| `04_signal_conditioning` | normalization, pre-emphasis and noise-gate negative controls |
| `05_atc_text_normalization` | WER rules plus merged display formatting for FL/RWY/frequencies |
| `06_hotword_callsign_playground` | callsign lexicons, Whisper prompts, CTC hotwords and numeric stress cases |
| `07_silero_vad_playground` | merged Silero/energy VAD sweeps and VAD/no-VAD ablations |
| `08_whisper_playground` | multiple base/fine-tuned Whisper checkpoints and decoding knobs |
| `09_directory_batch_transcription` | recursive WAV manifests, retries and failure-aware batch ASR |
| `10_voxtral_playground` | `pphilip/voxtral-3B-atc-transcribe`, prompts and memory estimates |
| `11_wav2vec2_xlsr_playground` | Wav2Vec2/XLS-R CTC checkpoints, vocab and optional KenLM decoding |
| `12_cross_family_benchmark` | one result schema for Whisper, CTC, NeMo and audio-language models |
| `13_fastconformer_playground` | `niclaswue/youtube-atc-fastconformer`, RNNT/CTC and NeMo batch flow |
| `14_metrics_sdi` | corpus WER/CER, S/D/I, alignment edge cases and run tables |
| `15_alignment_error_taxonomy` | callsign/number/command/runway/flight-level error categories |
| `16_normalization_ablation` | leave-one-rule-out normalization measurements |
| `17_latency_rtf_profiling` | load time, inference time, RTF, throughput and memory notes |
| `18_parakeet_canary_playground` | Parakeet base/fine-tune and Canary-Qwen LoRA inspection |

## Models added to the playgrounds

- Whisper: Jacktol, tclin ATCOSIM, youngsangroh small/large, jlvdoorn medium, and fjmgAI.
- CTC: Wav2Vec2 Large, XLS-R 300M, ATCOSIM/combined variants, and the robust four-gram candidate.
- NeMo: FastConformer, Parakeet-TDT base/fine-tune, and Canary-Qwen LoRA.
- Audio-language model: Voxtral 3B ATC adaptation.

The reported WER notes came from the request and different model cards/splits. They are stored as caveated notes, not as results produced by this archive. Live Hugging Face access was unavailable while this update was made; the `inspect_*_cards.py` scripts refresh IDs, files, commits and card metadata when run with network access.

## Notebook rule

Every notebook has at least eight cells. Some are longer because related experiments were merged. Notebooks contain no stored output and no execution counts; a fresh run is required.

## Validate without model downloads

```bash
python validate_archive.py
python validate_archive.py --json archive_validation.json
```

The validator checks structure, Python syntax, notebook schema/cell counts, provenance text, Persian notes, absence of `artifacts/` directories, and absence of stored notebook results. It does not execute GPU inference.

## Provenance rule

Do not backdate these files or present unexecuted code as completed research. When a playground is run, keep the exact command, model revision, dataset split/manifest hash, environment, errors and result file together.
