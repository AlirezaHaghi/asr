# Audio services

`app.py` only validates uploads and exposes HTTP routes. `APP_MODE` selects one
of three backends with the same response schemas.

| Mode | Value | Behavior |
| --- | --- | --- |
| Demo | `APP_MODE=demo` | Looks up results by uploaded filename in `mock.py`; no model or API call. This is the default. |
| Remote | `APP_MODE=remote` | Sends audio to Gemini through Pydantic AI and caches transcriptions by filename and content hash in SQLite. |
| Local | `APP_MODE=local` | Runs independent GPU models for ASR, accent classification, and speaker similarity. |

Restart the API after changing a mode or model environment variable.

## Files

| File | Responsibility |
| --- | --- |
| `models.py` | Internal audio value object, public Pydantic results, and service error. |
| `settings.py` | Reads and validates environment configuration. |
| `service.py` | Chooses a backend and owns the in-memory reference recording. |
| `mock.py` | Loads demo filename/result pairs from `audio-samples/result.txt`. |
| `remote.py` | The complete, compact Pydantic AI + Gemini implementation. |
| `database.py` | Remote transcription cache. |
| `local.py` | Combines the three local results and manages temporary audio files. |
| `local_asr.py` | Lazy ATC Whisper placeholder. |
| `local_accent.py` | Lazy English-accent classifier placeholder. |
| `local_speaker.py` | Lazy speaker embedding/similarity placeholder. |

Each local file also has a module docstring containing its model link and its
output caveat. Optional imports occur inside cached loader functions, so demo
and remote modes do not need the local-model packages.

## Demo data

The frontend uploads a recording from `audio-samples/samples`. Only the file's
name is used in demo mode; its bytes may be empty. `mock.py` loads the matching
transcription directly from `audio-samples/result.txt`, so result updates do not
need to be copied into Python code. All WAV recordings in the samples directory
are valid demo inputs.

These recordings currently have no entry in `result.txt`, so they return the
explicit placeholder `Transcription is not available in the provided mock
data.` until a transcript is supplied:

- `DABI358_Freq_128.25   Modulation_AM   Date_2020-12-16   Time_00-11-54-064   CH_6   Bank_Scan Table J.wav`
- `MANSOUR41_1399-1-12_20-52-46_124.500000MHz_AM.wav`

The default reference is `demo-reference.wav`. Speaker candidates are:

- `demo-same-speaker.wav`
- `demo-different-speaker.wav`

Names that are neither in `result.txt` nor the samples directory return HTTP
404 and list all available sample filenames.
Speaker-verification fixtures remain separate because the supplied result file
contains transcriptions but no speaker-identity labels.

## Remote Gemini configuration

Direct Google mode uses the existing `GOOGLE_API_KEY` from `.env`:

```dotenv
APP_MODE=remote
REMOTE_PROVIDER=google
GOOGLE_API_KEY=replace-me
GEMINI_MODEL=gemini-3.5-flash
```

`GOOGLE_API_KEYS` accepts a comma-separated list for round-robin requests.
`GOOGLE_PROXY_URL` is optional and supports HTTP or SOCKS URLs. The existing Metis-compatible path remains
available without putting secrets in source code:

```dotenv
APP_MODE=remote
REMOTE_PROVIDER=metis
METIS_API_KEY=replace-me
METIS_BASE_URL=https://api.metisai.ir
```

## Local GPU installation

Local ML packages are intentionally absent from the base `pyproject.toml` and
`uv.lock`. Use a dedicated environment so its CUDA-specific packages do not
become base application dependencies:

```powershell
uv venv .venv-local --python 3.11
uv pip install --python .venv-local\Scripts\python.exe -e .
# Install torch + torchaudio using the command generated for this GPU/driver at:
# https://pytorch.org/get-started/locally/
uv pip install --python .venv-local\Scripts\python.exe transformers accelerate safetensors speechbrain
$env:APP_MODE = "local"
.venv-local\Scripts\python.exe -m uvicorn app:app --reload
```

Install FFmpeg on the host and make it available on `PATH` for compressed audio
decoding. Set `LOCAL_DEVICE=cuda` (default) or a specific device such as
`cuda:0`. GPU wheels must match the installed NVIDIA driver/CUDA environment.

### Local ASR: `local_asr.py`

Model: [`jacktol/whisper-large-v3-finetuned-for-ATC`](https://huggingface.co/jacktol/whisper-large-v3-finetuned-for-ATC)
(MIT model card). It is the replacement for the now-deprecated medium ATC
checkpoint and is fine-tuned for English pilot/controller radio speech. The
author reports 6.5% WER on its prepared test split; that is not an independent
benchmark. The 1.55B-parameter model needs substantial VRAM and must be tested
on the target radio channels. Whisper can hallucinate, so do not use unreviewed
output for safety-critical decisions. The training dataset card does not state
a dataset license; review the original UWB/ATCO2 terms before redistribution.

The placeholder deliberately returns no ASR confidence because generation
scores are not a calibrated probability.

### Accent: `local_accent.py`

Model: [`Jzuluaga/accent-id-commonaccent_ecapa`](https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa)
(MIT). It is a roughly 83 MB SpeechBrain ECAPA model trained on 16 kHz English
CommonAccent speech. The card reports 87% test accuracy. Its labels are:
`england`, `us`, `canada`, `australia`, `indian`, `scotland`, `ireland`,
`african`, `malaysia`, `newzealand`, `southatlandtic`, `bermuda`,
`philippines`, `hongkong`, `wales`, and `singapore`. The source checkpoint
misspells `southatlandtic`; the placeholder displays it as `south atlantic`.

These are dataset accent categories, not nationality, citizenship, or
ethnicity. The API returns only the selected accent label.

### Speaker similarity: `local_speaker.py`

Model: [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
(Apache-2.0). It produces ECAPA speaker embeddings trained on VoxCeleb1+2 and
compares them with cosine similarity. SpeechBrain's default decision threshold
is `0.25`, exposed as `LOCAL_SPEAKER_THRESHOLD`.

The API returns the raw `similarity` and `threshold`; it does not mislabel the
score as confidence. Calibrate the threshold on representative ATC same/different
pairs, microphones, noise, accents, and clip lengths. This model is not an
anti-spoofing or liveness detector.
