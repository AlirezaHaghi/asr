import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

try:
    import noisereduce as nr

    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

try:
    from scipy.signal import butter, filtfilt

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


ACCENT_LABELS = [
    "us",
    "england",
    "australia",
    "indian",
    "canada",
    "bermuda",
    "scotland",
    "african",
    "ireland",
    "newzealand",
    "wales",
    "malaysia",
    "philippines",
    "singapore",
    "hongkong",
    "southatlandtic",
]

TARGET_SR = 16_000
ATC_LOW_HZ = 300
ATC_HIGH_HZ = 3_400


def load_audio(path: str) -> tuple[np.ndarray, int]:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze().numpy(), sr


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(torch.from_numpy(audio).float()).numpy()


def bandpass_filter(
    audio: np.ndarray,
    sr: int,
    low_hz: float = ATC_LOW_HZ,
    high_hz: float = ATC_HIGH_HZ,
    order: int = 5,
) -> np.ndarray:
    if not SCIPY_AVAILABLE:
        return audio
    nyq = sr / 2.0
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, audio).astype(np.float32)


def spectral_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    if not NOISEREDUCE_AVAILABLE:
        return audio
    noise_frames = int(0.5 * sr)
    noise_clip = audio[:noise_frames] if len(audio) > noise_frames else audio
    return nr.reduce_noise(
        y=audio,
        y_noise=noise_clip,
        sr=sr,
        stationary=False,
        prop_decrease=0.85,
    ).astype(np.float32)


def rms_normalize(audio: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-9:
        return audio
    return (audio / rms * target_rms).astype(np.float32)


def preprocess_atc_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = load_audio(path)
    audio = resample(audio, sr, TARGET_SR)
    audio = bandpass_filter(audio, TARGET_SR)
    audio = spectral_denoise(audio, TARGET_SR)
    audio = rms_normalize(audio)
    return audio, TARGET_SR


def save_temp_wav(audio: np.ndarray, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    return tmp.name


_xlsr_classifier = None
_ecapa_classifier = None


def get_xlsr_classifier(device: str = "cpu"):
    global _xlsr_classifier
    if _xlsr_classifier is not None:
        return _xlsr_classifier
    from speechbrain.pretrained.interfaces import foreign_class

    _xlsr_classifier = foreign_class(
        source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        run_opts={"device": device},
    )
    return _xlsr_classifier


def get_ecapa_classifier(device: str = "cpu"):
    global _ecapa_classifier
    if _ecapa_classifier is not None:
        return _ecapa_classifier
    from speechbrain.pretrained import EncoderClassifier

    _ecapa_classifier = EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_ecapa",
        savedir="pretrained_models/accent-id-commonaccent_ecapa",
        run_opts={"device": device},
    )
    return _ecapa_classifier


def _build_scores(out_prob, score, text_lab, clf) -> dict:
    probs = torch.softmax(out_prob, dim=-1).squeeze().tolist()
    all_scores: dict = {}
    try:
        label_encoder = clf.hparams.label_encoder
        for i, label in enumerate(label_encoder.ind2lab.values()):
            all_scores[label] = round(float(probs[i]) if i < len(probs) else 0.0, 4)
    except Exception:
        all_scores = {text_lab[0]: round(float(score.item()), 4)}
    return {
        "accent": text_lab[0],
        "confidence": round(float(score.item()), 4),
        "all_scores": dict(sorted(all_scores.items(), key=lambda x: -x[1])),
    }


def classify_accent_xlsr(
    audio_path: str,
    device: str = "cpu",
    skip_preprocessing: bool = False,
) -> dict:
    if skip_preprocessing:
        tmp_path = audio_path
    else:
        audio, sr = preprocess_atc_audio(audio_path)
        tmp_path = save_temp_wav(audio, sr)
    try:
        clf = get_xlsr_classifier(device)
        out_prob, score, index, text_lab = clf.classify_file(tmp_path)
        result = _build_scores(out_prob, score, text_lab, clf)
    finally:
        if not skip_preprocessing and tmp_path != audio_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    result["model"] = "Jzuluaga/accent-id-commonaccent_xlsr-en-english"
    return result


def classify_accent_ecapa(
    audio_path: str,
    device: str = "cpu",
    skip_preprocessing: bool = False,
) -> dict:
    if skip_preprocessing:
        tmp_path = audio_path
    else:
        audio, sr = preprocess_atc_audio(audio_path)
        tmp_path = save_temp_wav(audio, sr)
    try:
        clf = get_ecapa_classifier(device)
        out_prob, score, index, text_lab = clf.classify_file(tmp_path)
        result = _build_scores(out_prob, score, text_lab, clf)
    finally:
        if not skip_preprocessing and tmp_path != audio_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    result["model"] = "Jzuluaga/accent-id-commonaccent_ecapa"
    return result


def classify_accent(
    audio_path: str,
    model: str = "xlsr",
    device: Optional[str] = None,
    skip_preprocessing: bool = False,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model == "xlsr":
        return classify_accent_xlsr(audio_path, device, skip_preprocessing)
    if model == "ecapa":
        return classify_accent_ecapa(audio_path, device, skip_preprocessing)
    raise ValueError(f"Unknown model '{model}'. Choose 'xlsr' or 'ecapa'.")


def classify_batch(
    audio_paths: list[str],
    model: str = "xlsr",
    device: Optional[str] = None,
) -> list[dict]:
    results = []
    for path in audio_paths:
        try:
            res = classify_accent(path, model=model, device=device)
            res["file"] = path
            results.append(res)
        except Exception as e:
            results.append({"file": path, "error": str(e)})
    return results
