# ATC-ASR Pipeline

ASR for air traffic control using [`jacktol/whisper-medium.en-fine-tuned-for-ATC`](https://huggingface.co/jacktol/whisper-medium.en-fine-tuned-for-ATC).

## Layout

```
├── asr_engine.py
├── main.py
├── evaluate.py
├── text_normalizer.py
├── download_data.py
├── requirements.txt
└── colab/ATC_Benchmark_Colab.ipynb
```

## Install

```bash
pip install -r requirements.txt
```

## Usage

### Transcribe a folder

```bash
python main.py /path/to/audio_folder
```

Output goes to `audio_folder-transcripts/`.

### Benchmark on ATCO2-1h

```bash
python evaluate.py
```

Output in `eval_output/`:

| File | Contents |
|------|----------|
| `benchmark_results.json` | WER/CER + S/D/I for both runs |
| `predictions_run1.json` | Full predictions (beam=5) |
| `predictions_run2.json` | Full predictions (beam=1) |
| `error_analysis.txt` | 30+ categorized errors |

### Colab

Open `colab/ATC_Benchmark_Colab.ipynb`, set runtime to T4 GPU, run cells in order.

## Hardware

| Environment | Minimum |
|-------------|---------|
| Inference (medium) | 8 GB VRAM or CPU |
| Colab Free | T4 |
| RAM | 8 GB |

## Dataset

[ATCO2-test-set-1h](https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h)

## Reproducibility

```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
pip freeze > requirements_exact.txt
```
