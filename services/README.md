# Audio services

`app.py` validates uploads and exposes HTTP routes. `APP_MODE` selects one of
three backends with the same response schemas.

| Mode | Value | Behavior |
| --- | --- | --- |
| Demo | `APP_MODE=demo` | Looks up an uploaded filename in `mock.py`; no model or network call. This is the default. |
| Remote | `APP_MODE=remote` | Posts audio to the fixed legacy upstream and caches successful transcriptions in SQLite. |
| Local | `APP_MODE=local` | Runs separate local models for ASR, accent classification, and speaker similarity. |

Restart the API after changing environment settings.

## Files

| File | Responsibility |
| --- | --- |
| `models.py` | Internal audio object, public Pydantic results, and service error. |
| `settings.py` | Reads and validates environment configuration. |
| `service.py` | Chooses a backend and owns the in-memory reference recording. |
| `mock.py` | Loads demo filename/result pairs from `audio-samples/result.txt`. |
| `remote.py` | Calls the upstream HTTP transcription service. |
| `database.py` | Remote transcription cache. |
| `local.py` | Combines local results and manages temporary audio files. |
| `local_asr.py` | Lazy ATC Whisper loader and feature-flagged prompt hook. |
| `transcription_enrichment.py` | Surveillance sidecar validation, Whisper context preparation, and lazy local-LLM cleanup. |
| `local_accent.py` | Lazy English-accent classifier. |
| `local_speaker.py` | Lazy speaker embedding and similarity model. |

All AI imports occur inside cached local loader functions. Demo and remote mode
therefore require only the base dependencies installed by `uv sync`.

## Demo data

The frontend uploads a recording from `audio-samples/samples`. Only the name is
used in demo mode; `mock.py` reads its matching transcription from
`audio-samples/result.txt`.

Names present in the samples directory but absent from the result file return
`Transcription is not available in the provided mock data.` Unknown names
return HTTP 404 and list available fixtures. Speaker-verification fixtures are
separate because the transcription data has no identity labels.

## Remote mode

Remote mode uses plain `httpx`; it does not import or install an AI SDK or a
local model. The current legacy implementation keeps its fixed upstream values
in `remote.py`. Move those values to deployment configuration and secret
management, and rotate any committed credential, before production use.

```powershell
uv sync
$env:APP_MODE = "remote"
uv run uvicorn app:app --reload
```

## Local GPU mode

The project `local` extra selects the official PyTorch CUDA 12.8 index on
Windows and Linux:

```powershell
uv sync --extra local
$env:APP_MODE = "local"
$env:LOCAL_DEVICE = "cuda"
uv run --extra local uvicorn app:app --reload
```

Do not run this installation on a demo/remote-only host. It installs PyTorch,
Transformers, Accelerate, safetensors, SpeechBrain, torchaudio, and
bitsandbytes. Model weights are fetched lazily by the corresponding
`from_pretrained` call. Install FFmpeg separately for compressed audio.

### ATC ASR

Model:
[`jacktol/whisper-large-v3-finetuned-for-ATC`](https://huggingface.co/jacktol/whisper-large-v3-finetuned-for-ATC)
(MIT). It is a 1.55B-parameter Whisper large-v3 checkpoint fine-tuned for
English pilot/controller radio speech. The author reports 6.5% WER on its
prepared test split; this is not an independent benchmark. The endpoint does
not expose a confidence because generation scores are not calibrated
probabilities.

### Optional surveillance and LLM enrichment

Set `ENRICH_USING_SURVEILLANCE_DATA=true` to enable both stages. The pre-ASR
stage calls the single entry point as `enrich_transcription(audio, None)` and
turns its compact context into Whisper `prompt_ids`. The post-ASR stage calls
`enrich_transcription(audio, raw_text)` and returns its string result.

The `None` call is an internal phase marker necessitated by generation order:
the final transcript does not exist when decoder context must be prepared.
`local_asr.py` imports no other enrichment symbol.

Configuration:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENRICH_USING_SURVEILLANCE_DATA` | `false` | Enables both prompt injection and postprocessing. |
| `SURVEILLANCE_DATA_PATH` | beside audio path | SHA-256-bound central JSON or directory of per-audio sidecars; configure it for API uploads. |
| `LOCAL_REFINEMENT_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Text cleanup checkpoint. |
| `LOCAL_LLM_DEVICE` | value of `LOCAL_DEVICE` | Device used by the refiner. |

The application-specific sidecar schema and research basis are documented in
the root README. API uploads should always configure a server-side path; the
client-side audio directory is not available to the service. Each record must
contain the SHA-256 of the exact audio bytes. A missing sidecar silently skips
Whisper prompting; a malformed or mismatched sidecar skips it with a runtime
warning. The LLM may still clean obvious repetition. Missing local AI packages
produce HTTP 503 when the feature is explicitly enabled. Unexpected
LLM-generation errors retain the raw transcript.

The refiner uses deterministic generation. Its output is accepted only when it
removes repeated/filler tokens without deleting unique operational words, or
when its sole semantic change is an exact active spoken callsign form. It is
still a generative, text-only model and cannot verify the audio. The current API
returns the selected transcript only; it does not expose raw/enriched pairs or
an enrichment indicator. Extend that response/persistence contract before
enabling the feature in any workflow that requires a complete audit trail.

### Accent

Model:
[`Jzuluaga/accent-id-commonaccent_ecapa`](https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa)
(MIT). It is an approximately 83 MB SpeechBrain ECAPA classifier trained on
16 kHz English CommonAccent speech. Its labels are dataset accent categories,
not nationality, citizenship, or ethnicity.

### Speaker similarity

Model:
[`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
(Apache-2.0). It compares ECAPA embeddings with cosine similarity. The default
threshold is `LOCAL_SPEAKER_THRESHOLD=0.25`; calibrate it on representative ATC
channels. This model is not an anti-spoofing or liveness detector.
