"""Environment-backed settings for backend selection and model configuration."""

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


class AppMode(str, Enum):
    DEMO = "demo"
    REMOTE = "remote"
    LOCAL = "local"


@dataclass(frozen=True, slots=True)
class Settings:
    mode: AppMode
    gemini_model: str
    remote_provider: str
    google_api_keys: tuple[str, ...]
    metis_api_key: str | None
    metis_base_url: str
    google_proxy_url: str | None
    default_voice_sample: Path
    local_device: str
    local_speaker_threshold: float


def _mode() -> AppMode:
    raw = os.getenv("APP_MODE", AppMode.DEMO.value).strip().lower()
    print(raw)
    try:
        return AppMode(raw)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in AppMode)
        raise RuntimeError(f"APP_MODE must be one of: {choices}") from exc

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    key_list = os.getenv("GOOGLE_API_KEYS", "")
    keys = tuple(key.strip() for key in key_list.split(",") if key.strip())
    if not keys and os.getenv("GOOGLE_API_KEY"):
        keys = (os.environ["GOOGLE_API_KEY"],)

    setting = Settings(
        mode=_mode(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        remote_provider=os.getenv("REMOTE_PROVIDER", "google").strip().lower(),
        google_api_keys=keys,
        metis_api_key=os.getenv("METIS_API_KEY"),
        metis_base_url=os.getenv("METIS_BASE_URL", "https://api.metisai.ir"),
        google_proxy_url=os.getenv("GOOGLE_PROXY_URL"),
        default_voice_sample=Path(
            os.getenv("DEFAULT_VOICE_SAMPLE_PATH", "default_sample.wav")
        ),
        local_device=os.getenv("LOCAL_DEVICE", "cuda"),
        local_speaker_threshold=float(os.getenv("LOCAL_SPEAKER_THRESHOLD", "0.25")),
    )
    print(setting)
    return setting