# Delivered evidence and retrospective playgrounds

## Directly connected to the delivered notebook/report

| Delivered feature | Playground areas |
|---|---|
| ATCO2 one-hour loading and schema | `01`, `02` |
| 16 kHz mono audio assumption | `03` |
| optional Silero VAD and VAD ablation | `07` |
| Whisper ATC checkpoints and beam 1/5 | `08` |
| ATC normalizer and display forms | `05`, `16` |
| directory-oriented transcription exploration | `09` |
| XLS-R CTC baseline | `11` |
| WER/CER and S/D/I | `14` |
| word alignment and error categories | `15` |
| duration/inference timing | `17` |
| configs, cache fingerprints and result comparison | `12` |

## Retrospective extensions based on the manual/review request

- `01` broadens dataset inspection to combined/local ATC corpora.
- `02` makes split overlap and leakage checks explicit.
- `04` keeps signal conditioning as a negative-control study.
- `06` prototypes callsign, number and hotword recommendations.
- `10`, `11`, `13`, and `18` add open-source non-Whisper playgrounds.
- `12` provides a cross-family result schema instead of pretending every model shares one loader.

## Model identifiers carried into code/catalogs

- `jacktol/whisper-large-v3-finetuned-for-ATC`
- `tclin/whisper-large-v3-turbo-atcosim`
- `youngsangroh/whisper-small-atco2-atcosim`
- `youngsangroh/whisper-large-atco2-atcosim`
- `jlvdoorn/whisper-medium-atco2`
- `Jzuluaga/wav2vec2-large-960h-lv60-self-en-atc-uwb-atcc-and-atcosim`
- `Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim`
- `niclaswue/youtube-atc-fastconformer`
- `qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC`
- `pphilip/voxtral-3B-atc-transcribe`
- `suideepmax/canary-qwen-2.5b-atc-lora`
- `ThaiVanPhat95/wav2vec2-robust-uwb-atcosim-supcon-hybrid-4gram`

These identifiers were supplied by the requester or already evidenced in the delivered project. Live Hugging Face browsing failed during this update, so scripts label card figures as unverified notes and provide runtime card inspection. Do not compare ATCOSIM, UWB-ATCC, Jacktol combined splits and untouched ATCO2 as if they were the same benchmark.
