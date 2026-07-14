"""Heuristic confidence estimation for local ASR transcriptions.

The local ASR backend does not expose a native confidence score. This module
derives an approximate one from surface-level signals — chiefly how many
words/characters were produced relative to the audio's duration, compared
against baseline human speech-rate statistics — plus a few sanity checks
(degenerate/empty output, pathological repetition, implausibly short audio).

Design notes
------------
The module is intentionally structured so new signals can be added without
touching the public entry point:

1. Implement a class deriving from ``ConfidenceSignal`` with a ``score()``
   method returning a value in ``[0, 1]``.
2. Add an instance of it (with a weight) to ``DEFAULT_SIGNALS``, or pass a
   custom sequence of signals into ``ConfidenceEstimator``.

The only function other modules should depend on is
``estimate_transcription_confidence``; everything else is an implementation
detail that is free to evolve.
"""

from __future__ import annotations

import io
import math
import re
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from .models import AudioInput

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Floor applied to any duration estimate so we never divide by zero and never
# let a near-zero denominator blow the rate-based signals up to absurd
# values.
_MINIMUM_DURATION_SECONDS = 0.05

# Rough, English-conversational-speech baselines. These are deliberately
# generous (wide std) since they need to tolerate multiple languages,
# speaking styles, and pauses without being overly punitive.
_DEFAULT_WORDS_PER_SECOND_MEAN = 2.5   # ~150 words per minute
_DEFAULT_WORDS_PER_SECOND_STD = 0.9
_DEFAULT_CHARS_PER_SECOND_MEAN = 14.0
_DEFAULT_CHARS_PER_SECOND_STD = 5.0

# Assumed average encoded bitrate (bits/second) used as a last-resort
# duration estimate when we can't parse a proper header and have no
# metadata library available. WAV is intentionally absent: it is handled by
# a real header parse instead.
_ASSUMED_BITRATE_BITS_PER_SECOND = {
    "audio/mpeg": 128_000,
    "audio/mp4": 128_000,
    "audio/x-m4a": 128_000,
    "audio/flac": 800_000,
}
_DEFAULT_ASSUMED_BITRATE_BITS_PER_SECOND = 128_000

_WORD_PATTERN = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------------


def _duration_from_wave_header(audio: AudioInput) -> float | None:
    """Exact duration for uncompressed WAV via its header."""
    if audio.media_type not in ("audio/wav", "audio/x-wav"):
        return None
    try:
        with wave.open(io.BytesIO(audio.content)) as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
    except (wave.Error, EOFError, ValueError):
        return None
    if frame_rate <= 0:
        return None
    return frame_count / frame_rate


def _duration_from_mutagen(audio: AudioInput) -> float | None:
    """Best-effort duration via ``mutagen``, if it happens to be installed."""
    try:
        from mutagen import File as MutagenFile  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        parsed = MutagenFile(io.BytesIO(audio.content))
    except Exception:
        return None
    if parsed is None or getattr(parsed, "info", None) is None:
        return None
    length = getattr(parsed.info, "length", None)
    if not length or length <= 0:
        return None
    return float(length)


def _duration_from_bitrate_heuristic(audio: AudioInput) -> float:
    """Last-resort estimate: file size divided by an assumed bitrate.

    This is intentionally crude — it exists only so the pipeline never
    fails outright when it can't determine a real duration.
    """
    bitrate = _ASSUMED_BITRATE_BITS_PER_SECOND.get(
        audio.media_type, _DEFAULT_ASSUMED_BITRATE_BITS_PER_SECOND
    )
    size_bits = len(audio.content) * 8
    return size_bits / bitrate


def estimate_audio_duration_seconds(audio: AudioInput) -> float:
    """Estimate the audio's duration in seconds, trying increasingly rough
    strategies until one succeeds.
    """
    for extractor in (_duration_from_wave_header, _duration_from_mutagen):
        duration = extractor(audio)
        if duration is not None and duration > 0:
            return duration
    return max(_duration_from_bitrate_heuristic(audio), _MINIMUM_DURATION_SECONDS)


# ---------------------------------------------------------------------------
# Transcript metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeechMetrics:
    """Derived, read-only measurements used by every confidence signal."""

    duration_seconds: float
    tokens: tuple[str, ...]

    @property
    def word_count(self) -> int:
        return len(self.tokens)

    @property
    def char_count(self) -> int:
        return sum(len(token) for token in self.tokens)

    @property
    def words_per_second(self) -> float:
        return self.word_count / self.duration_seconds

    @property
    def chars_per_second(self) -> float:
        return self.char_count / self.duration_seconds

    @property
    def unique_token_ratio(self) -> float:
        if not self.tokens:
            return 1.0
        return len(set(self.tokens)) / len(self.tokens)


def _build_metrics(transcription: str, duration_seconds: float) -> SpeechMetrics:
    duration = max(duration_seconds, _MINIMUM_DURATION_SECONDS)
    tokens = tuple(match.group() for match in _WORD_PATTERN.finditer(transcription or ""))
    return SpeechMetrics(duration_seconds=duration, tokens=tokens)


# ---------------------------------------------------------------------------
# Baseline speech-rate statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeechRateBaseline:
    """A (mean, std) pair per rate metric, used as the "expected" profile.

    Extending to per-language or per-domain baselines later is just a
    matter of constructing additional ``SpeechRateBaseline`` instances and
    picking one based on context (e.g. detected language) before building
    the signals.
    """

    words_per_second_mean: float = _DEFAULT_WORDS_PER_SECOND_MEAN
    words_per_second_std: float = _DEFAULT_WORDS_PER_SECOND_STD
    chars_per_second_mean: float = _DEFAULT_CHARS_PER_SECOND_MEAN
    chars_per_second_std: float = _DEFAULT_CHARS_PER_SECOND_STD


DEFAULT_BASELINE = SpeechRateBaseline()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _gaussian_score(value: float, mean: float, std: float) -> float:
    """Bell-curve score: 1.0 at the mean, decaying towards 0 the further
    ``value`` strays from it, in units of ``std``.
    """
    if std <= 0:
        return 1.0 if math.isclose(value, mean) else 0.0
    z = (value - mean) / std
    return _clamp01(math.exp(-0.5 * z * z))


# ---------------------------------------------------------------------------
# Individual confidence signals
# ---------------------------------------------------------------------------


class ConfidenceSignal(ABC):
    """One scoring strategy contributing to the overall confidence."""

    name: str
    weight: float

    @abstractmethod
    def score(self, metrics: SpeechMetrics) -> float:
        """Return a value in ``[0, 1]``; higher means more confident."""


class WordRateSignal(ConfidenceSignal):
    """How close the words/second rate is to a typical speech baseline."""

    name = "word_rate"

    def __init__(self, baseline: SpeechRateBaseline = DEFAULT_BASELINE, weight: float = 0.30):
        self.baseline = baseline
        self.weight = weight

    def score(self, metrics: SpeechMetrics) -> float:
        if metrics.word_count == 0:
            # Handled by NonEmptySignal; don't double-penalize here.
            return 1.0
        return _gaussian_score(
            metrics.words_per_second,
            self.baseline.words_per_second_mean,
            self.baseline.words_per_second_std,
        )


class CharRateSignal(ConfidenceSignal):
    """How close the characters/second rate is to a typical speech baseline."""

    name = "char_rate"

    def __init__(self, baseline: SpeechRateBaseline = DEFAULT_BASELINE, weight: float = 0.25):
        self.baseline = baseline
        self.weight = weight

    def score(self, metrics: SpeechMetrics) -> float:
        if metrics.char_count == 0:
            return 1.0
        return _gaussian_score(
            metrics.chars_per_second,
            self.baseline.chars_per_second_mean,
            self.baseline.chars_per_second_std,
        )


class RepetitionSignal(ConfidenceSignal):
    """Penalizes pathological token repetition (a common ASR hallucination
    failure mode: the model gets "stuck" looping the same word/phrase).
    """

    name = "repetition"

    def __init__(
        self,
        minimum_tokens: int = 8,
        healthy_unique_ratio: float = 0.4,
        weight: float = 0.15,
    ):
        self.minimum_tokens = minimum_tokens
        self.healthy_unique_ratio = healthy_unique_ratio
        self.weight = weight

    def score(self, metrics: SpeechMetrics) -> float:
        if metrics.word_count < self.minimum_tokens:
            # Too short a transcript for repetition ratio to be meaningful.
            return 1.0
        ratio = metrics.unique_token_ratio
        if ratio >= self.healthy_unique_ratio:
            return 1.0
        return _clamp01(ratio / self.healthy_unique_ratio)


class NonEmptySignal(ConfidenceSignal):
    """Sanity-checks empty transcriptions against audio duration.

    An empty transcript is fine for near-silent/very short audio, but
    suspicious for anything long enough to plausibly contain speech.
    """

    name = "non_empty"

    def __init__(self, minimum_duration_for_speech: float = 0.75, weight: float = 0.15):
        self.minimum_duration_for_speech = minimum_duration_for_speech
        self.weight = weight

    def score(self, metrics: SpeechMetrics) -> float:
        if metrics.word_count > 0:
            return 1.0
        if metrics.duration_seconds <= self.minimum_duration_for_speech:
            return 1.0
        return 0.2


class DurationSanitySignal(ConfidenceSignal):
    """Flags transcripts produced from implausibly short audio clips."""

    name = "duration_sanity"

    def __init__(self, implausibly_short_seconds: float = 0.2, weight: float = 0.15):
        self.implausibly_short_seconds = implausibly_short_seconds
        self.weight = weight

    def score(self, metrics: SpeechMetrics) -> float:
        if metrics.duration_seconds <= self.implausibly_short_seconds and metrics.word_count > 0:
            return 0.1
        return 1.0


DEFAULT_SIGNALS: tuple[ConfidenceSignal, ...] = (
    WordRateSignal(),
    CharRateSignal(),
    RepetitionSignal(),
    NonEmptySignal(),
    DurationSanitySignal(),
)


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------


class ConfidenceEstimator:
    """Combines a set of ``ConfidenceSignal``s into a single score via a
    weighted average.
    """

    def __init__(self, signals: Sequence[ConfidenceSignal] = DEFAULT_SIGNALS):
        if not signals:
            raise ValueError("ConfidenceEstimator requires at least one signal")
        self._signals = tuple(signals)

    def estimate(self, metrics: SpeechMetrics) -> float:
        total_weight = sum(signal.weight for signal in self._signals)
        if total_weight <= 0:
            raise ValueError("Signal weights must sum to a positive number")
        weighted_sum = sum(
            signal.weight * _clamp01(signal.score(metrics)) for signal in self._signals
        )
        return _clamp01(weighted_sum / total_weight)

def estimate_transcription_confidence(transcription: str, audio: AudioInput) -> float:
    """Estimate ASR confidence for ``transcription`` given the ``audio`` it
    was produced from.

    This is the single supported entry point into this module — everything
    above is free to be extended or restructured without affecting callers.

    Returns a value in ``[0, 1]``, where higher means more confident.
    """
    duration_seconds = estimate_audio_duration_seconds(audio)
    metrics = _build_metrics(transcription, duration_seconds)
    return ConfidenceEstimator().estimate(metrics)