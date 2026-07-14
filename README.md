# ATC ASR pipeline

FastAPI service and evaluation tools for air-traffic-control speech. The API
supports lightweight demo/remote operation and an optional local GPU stack.

## Lightweight install: demo or remote

```powershell
uv sync
uv run uvicorn app:app --reload
```

Set `APP_MODE=demo` (the default) or `APP_MODE=remote` in `.env`. A plain
`uv sync` installs only the web, HTTP, validation, and database dependencies.
It does **not** install PyTorch, Transformers, SpeechBrain, bitsandbytes, or the
benchmark packages.

| Mode | Behavior |
| --- | --- |
| `demo` | Reads filename-based fixtures from `audio-samples/result.txt`; no model or API call. |
| `remote` | Sends audio to the upstream HTTP transcription service and caches its response. |
| `local` | Runs ATC Whisper, accent classification, speaker comparison, and optional transcript enrichment on local hardware. |

See [services/README.md](services/README.md) for backend details and model
limitations.

## Local NVIDIA GPU install

The `local` extra is intentionally separate and is configured for the official
PyTorch CUDA 12.8 wheel index on Windows and Linux:

```powershell
uv sync --extra local
$env:APP_MODE = "local"
uv run --extra local uvicorn app:app --reload
```

Run those commands only on the local-model host. `uv sync --extra local`
downloads the AI runtime wheels, and the first local request downloads model
checkpoints from Hugging Face. The default demo/remote command never selects
that extra.

If the deployment driver requires another PyTorch build, change the explicit
`pytorch-cu128` index and the matching `tool.uv.sources` entries in
`pyproject.toml` to an index supported by that host, then run `uv lock` there.
The official [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
lists the available CUDA indexes. FFmpeg must also be on `PATH` for compressed
audio.

## Surveillance-aware enrichment

Enrichment is off by default. Enable both the Whisper preprocessing prompt and
the local-LLM postprocessor with:

```dotenv
APP_MODE=local
LOCAL_DEVICE=cuda
ENRICH_USING_SURVEILLANCE_DATA=true
SURVEILLANCE_DATA_PATH=C:\data\atc-surveillance
LOCAL_REFINEMENT_MODEL=Qwen/Qwen2.5-7B-Instruct
LOCAL_LLM_DEVICE=cuda
```

When the flag is false, local Whisper runs with the original generation
arguments and returns its original stripped text. When it is true:

1. The active callsigns, spoken callsign forms, waypoints, and frequency are
   converted to a compact transcript-like prompt and passed to Whisper through
   `prompt_ids`.
2. A 4-bit NF4
   [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
   text-only pass is instructed to remove obvious ASR repetition and reconcile
   a callsign with an active surveillance candidate. The acceptance guard
   allows only repetition/filler removal or a callsign change to an exact
   active spoken form; all other word changes fall back to raw ASR text.

For API uploads, set `SURVEILLANCE_DATA_PATH` to one central JSON file or a
server-side directory. In a directory, context for `sample.wav` may be named either
`sample.wav.surveillance.json` or `sample.surveillance.json`. A central file is
keyed by the server-sanitized upload basename (ordinary filenames such as
`sample.wav` are unchanged):

```json
{
  "schema_version": 1,
  "recordings": {
    "sample.wav": {
      "audio_sha256": "b1f5e8f65cca0c9e1f9f6de47c4c136b11f12f7187f35a50f9f0ce25f1f47f56",
      "segments": [
        {
          "start_seconds": 0.0,
          "end_seconds": 4.25,
          "callsigns": [
            {
              "icao": "SWR2689",
              "spoken_forms": [
                "swiss two six eight nine",
                "swiss six eight nine"
              ]
            }
          ],
          "waypoints": ["BIGLU"],
          "frequency_mhz": 127.35
        }
      ]
    }
  }
}
```

This is an application-specific adapter schema, not native ATCO2 XML. Generate
it from an authorized ADS-B/Mode-S source and bind each entry to the exact audio
bytes with lowercase SHA-256. For example, PowerShell's
`(Get-FileHash sample.wav -Algorithm SHA256).Hash.ToLower()` produces the value.
The service rejects a filename match whose hash differs, so renaming unrelated
audio cannot select another recording's context.

The application does not query live surveillance services and does not filter
the JSON segments itself. The producer must first filter and relevance-order
callsigns by the utterance timestamp and location. The service reads at most
the first 64 segments, 32 callsigns, three spoken forms per callsign, and 32
waypoints, then caps the Whisper prompt at 128 tokens. Do not pass an
airport-wide or corpus-wide callsign list. Supply proper airline telephony
variants in `spoken_forms`; an ICAO value without spoken forms is not inserted
into the Whisper prompt.

### Why a separate sidecar is required

ATCO2 research uses timestamp and location to retrieve likely aircraft, applies
WFST/lattice callsign boosting, then performs NER and callsign correlation. This
project is an unvalidated adaptation: it uses soft Whisper decoder context and
a general text LLM, so the papers' reported improvements do not transfer to
this implementation. See the primary papers on
[contextual callsign boosting](https://arxiv.org/abs/2202.03725) and
[air-surveillance-assisted semi-supervised ASR](https://arxiv.org/abs/2104.03643).
The official
[ATCO2 conversion and callsign-expansion tools](https://github.com/idiap/atco2-corpus)
can be used as the source for an adapter into the JSON schema above.

The Hugging Face
[`Jzuluaga/atco2_corpus_1h`](https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h)
card mentions surveillance metadata in the full ATCO2 corpus, but the hosted
1-hour schema exposes only audio, text, IDs, and segment timing. The
[`jacktol/ATC-ASR-Dataset`](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset)
used by the current checkpoint exposes only ID, audio, and text. Neither hosted
table can supply active callsigns at inference time, hence the explicit JSON
input above.

Hugging Face documents Whisper `prompt_ids` as a way to bias custom vocabulary
and proper nouns. It is a soft bias, not a constraint. For audio requiring
Whisper long-form segmentation, the default prompt affects the first segment
only; the local checkpoint is intended for short, distinct ATC transmissions.
The Qwen stage sees text and sidecar context, not acoustic features, so it
cannot listen back to verify the transcript. Enriched output still requires
human review and must not be used as an unreviewed safety decision.

## Local model memory choice

The refiner is Qwen2.5-7B-Instruct (7.61B parameters, Apache-2.0) loaded through
bitsandbytes 4-bit NF4. The standard checkpoint download contains about
14.2 GiB of BF16 weights, but it is quantized while loading rather than kept in
BF16 GPU memory. Hugging Face documents 4-bit loading as compressing the linear
layers to 4-bit storage; this leaves substantially more headroom beside the
roughly 2.9 GiB FP16 Whisper parameter payload on a 20 GiB GPU. This is a
capacity estimate, not a measured peak. Actual memory depends on audio length,
prompt length, driver, quantization buffers, and allocator behavior, so verify
it on the deployment GPU. See the official
[bitsandbytes integration guide](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes).

## Benchmark and legacy scripts

Benchmark-only libraries are another optional extra:

```powershell
# Evaluation uses the local model and benchmark tools.
uv sync --extra local --extra benchmark
uv run --extra local --extra benchmark python evaluate.py

# Recompute metrics without running local inference.
uv run --extra benchmark python renormalize.py
```

Common commands:

```powershell
# Transcribe a folder with the legacy script.
uv run --extra local --extra benchmark python main.py C:\path\to\audio

# Download the evaluation dataset on a machine intended to hold it.
uv run --extra benchmark python download_data.py
```

Benchmark output is written under `eval_output/` or `atc_asr_output/`, depending
on the script. The Colab workflow is in `colab/ATC_Benchmark_Colab.ipynb`.
