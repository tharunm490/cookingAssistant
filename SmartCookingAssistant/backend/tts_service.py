from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import re

from gtts import gTTS
from gtts.tts import gTTSError

try:
    import pyttsx3
except Exception:  # pragma: no cover - optional fallback
    pyttsx3 = None

from .config import settings
from .utils import ensure_directory, join_recipe_steps, slugify


class TTSService:
    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = ensure_directory(output_dir or settings.audio_dir)

    def _sanitize_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").replace("•", ".").replace("\n", " ")).strip()
        return cleaned

    def _compact_narration(self, recipe: dict) -> str:
        name = self._sanitize_text(str(recipe.get("recipe_name") or "This recipe"))
        ingredients = recipe.get("ingredients_with_measurements") or []
        steps = recipe.get("steps") or []

        ingredient_lines: list[str] = []
        for item in ingredients[:6]:
            if isinstance(item, dict):
                ingredient = self._sanitize_text(str(item.get("ingredient", "")))
                measurement = self._sanitize_text(str(item.get("measurement", "")))
                if ingredient:
                    ingredient_lines.append(f"{measurement} {ingredient}".strip())

        narration_parts = [f"Recipe name: {name}."]
        if ingredient_lines:
            narration_parts.append("Main ingredients: " + ", ".join(ingredient_lines) + ".")
        if steps:
            short_steps = [self._sanitize_text(str(step)) for step in steps[:4]]
            narration_parts.append("Quick steps: " + " ".join(short_steps) + ".")
        return self._sanitize_text(" ".join(part for part in narration_parts if part.strip()))

    def build_narration(self, recipe: dict) -> str:
        name = self._sanitize_text(str(recipe.get("recipe_name") or "This recipe"))
        ingredients = recipe.get("ingredients_with_measurements") or []
        steps = recipe.get("steps") or []

        ingredient_lines = []
        for item in ingredients[:8]:
            if isinstance(item, dict):
                ingredient = self._sanitize_text(str(item.get("ingredient", "")))
                measurement = self._sanitize_text(str(item.get("measurement", "")))
                if ingredient:
                    ingredient_lines.append(f"{measurement} {ingredient}".strip())

        narration_parts = [f"Recipe name: {name}."]
        if ingredient_lines:
            narration_parts.append("Ingredients: " + ", ".join(ingredient_lines) + ".")
        if steps:
            short_steps = [self._sanitize_text(str(step)) for step in steps[:5]]
            narration_parts.append("Steps: " + " ".join(short_steps) + ".")

        narration = self._sanitize_text(" ".join(part for part in narration_parts if part.strip()))
        if len(narration) > 900:
            return self._compact_narration(recipe)
        return narration

    def create_audio(self, recipe: dict, language: str = "en") -> dict[str, str]:
        narration = self.build_narration(recipe)
        if not narration:
            narration = self._compact_narration(recipe)
        base_name = f"{slugify(recipe.get('recipe_name', 'recipe'))}-{uuid4().hex[:8]}"
        file_path = self.output_dir / f"{base_name}.mp3"
        lang = (language or "en").strip().lower()
        if lang not in {"en", "hi", "ta", "te", "kn"}:
            lang = "en"
        try:
            gTTS(text=narration, lang=lang, slow=False).save(str(file_path))
        except (gTTSError, ValueError):
            # Retry once with a shorter narration for transient network interruptions or long text.
            try:
                compact_narration = self._compact_narration(recipe)
                gTTS(text=compact_narration, lang=lang, slow=False).save(str(file_path))
            except (gTTSError, ValueError) as exc:
                # Offline fallback for Windows environments or temporary network failures.
                wav_path = self.output_dir / f"{base_name}.wav"
                if pyttsx3 is not None:
                    try:
                        engine = pyttsx3.init()
                        engine.save_to_file(self._compact_narration(recipe), str(wav_path))
                        engine.runAndWait()
                        file_path = wav_path
                    except Exception as fallback_exc:
                        raise RuntimeError("Text-to-speech service is unavailable. Please retry.") from fallback_exc
                else:
                    raise RuntimeError("Text-to-speech service is unavailable. Please retry.") from exc
        return {
            "audio_path": str(file_path),
            "audio_url": f"{settings.audio_route}/{file_path.name}",
        }
