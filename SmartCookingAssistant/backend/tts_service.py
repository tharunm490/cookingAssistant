from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from gtts import gTTS
from gtts.tts import gTTSError

from .config import settings
from .utils import ensure_directory, join_recipe_steps, slugify


class TTSService:
    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = ensure_directory(output_dir or settings.audio_dir)

    def build_narration(self, recipe: dict) -> str:
        name = recipe.get("recipe_name", "This recipe")
        ingredients = recipe.get("ingredients_with_measurements") or []
        steps = recipe.get("steps") or []

        ingredient_lines = []
        for item in ingredients:
            if isinstance(item, dict):
                ingredient = item.get("ingredient", "")
                measurement = item.get("measurement", "")
                ingredient_lines.append(f"{measurement} {ingredient}".strip())

        narration_parts = [f"Recipe name: {name}."]
        if ingredient_lines:
            narration_parts.append("Ingredients: " + ", ".join(ingredient_lines) + ".")
        if steps:
            narration_parts.append("Steps: " + join_recipe_steps([str(step) for step in steps]) + ".")
        return " ".join(part for part in narration_parts if part.strip())

    def create_audio(self, recipe: dict, language: str = "en") -> dict[str, str]:
        narration = self.build_narration(recipe)
        filename = f"{slugify(recipe.get('recipe_name', 'recipe'))}-{uuid4().hex[:8]}.mp3"
        file_path = self.output_dir / filename
        lang = (language or "en").strip().lower()
        if lang not in {"en", "hi", "ta", "te", "kn"}:
            lang = "en"
        try:
            gTTS(text=narration, lang=lang, slow=False).save(str(file_path))
        except gTTSError:
            # Retry once for transient network interruptions.
            try:
                gTTS(text=narration, lang=lang, slow=False).save(str(file_path))
            except gTTSError as exc:
                raise RuntimeError("Text-to-speech service is unavailable. Please retry.") from exc
        return {
            "audio_path": str(file_path),
            "audio_url": f"{settings.audio_route}/{filename}",
        }
