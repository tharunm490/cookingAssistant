from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _default_audio_dir() -> Path:
    configured = os.getenv("SCA_AUDIO_DIR")
    if configured:
        return Path(configured)

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "SmartCookingAssistant" / "generated_audio"

    return PROJECT_DIR.parent / "SmartCookingAssistantAudio"


GENERATED_AUDIO_DIR = _default_audio_dir()
GENERATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _load_local_env() -> None:
    env_path = PROJECT_DIR / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#") or "=" not in cleaned:
            continue
        key, value = cleaned.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()


@dataclass(frozen=True)
class Settings:
    hf_token: str | None = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    clip_model_name: str = "openai/clip-vit-base-patch32"
    recipe_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_model: bool = _env_bool("USE_MODEL", False)
    audio_dir: Path = GENERATED_AUDIO_DIR
    audio_route: str = "/audio"
    max_detected_ingredients: int = 5
    min_confidence: float = 0.12
    cors_origins: tuple[str, ...] = (
        "*",
    )


settings = Settings()
