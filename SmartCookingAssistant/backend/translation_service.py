from __future__ import annotations

import logging
from typing import Any

from deep_translator import GoogleTranslator

LANGUAGE_TO_FLORES: dict[str, str] = {
    "hi": "hi",
    "ta": "ta",
    "te": "te",
    "kn": "kn",
}

logger = logging.getLogger(__name__)


class TranslationService:
    def __init__(self) -> None:
        self._translator_cache: dict[str, GoogleTranslator] = {}

    def _get_translator(self, target_lang: str) -> GoogleTranslator:
        translator = self._translator_cache.get(target_lang)
        if translator is None:
            translator = GoogleTranslator(source="en", target=target_lang)
            self._translator_cache[target_lang] = translator
        return translator

    def translate(self, text: str, target_lang: str) -> str:
        clean_text = (text or "").strip()
        if not clean_text:
            return ""

        translator = self._get_translator(target_lang)
        translated = translator.translate(clean_text)
        return translated.strip() if translated else clean_text


_translator = TranslationService()


def translate_to_hindi(text: str) -> str:
    """Translate English text to Hindi."""
    try:
        return _translator.translate(text=text, target_lang="hi")
    except Exception as exc:
        # Keep pipeline resilient if translation model is unavailable.
        logger.warning("Hindi translation failed, returning original text: %s", exc)
        return text


def translate_text(text: str, language: str) -> str:
    lang = (language or "").strip().lower()
    if lang == "en" or not lang:
        return text

    target_lang = LANGUAGE_TO_FLORES.get(lang)
    if not target_lang:
        return text

    try:
        return _translator.translate(text=text, target_lang=target_lang)
    except Exception as exc:
        logger.warning("Translation failed for language '%s', returning original text: %s", lang, exc)
        return text


def _translate_value(value: Any, language: str) -> Any:
    if isinstance(value, str):
        return translate_text(value, language)
    if isinstance(value, list):
        return [_translate_value(item, language) for item in value]
    if isinstance(value, dict):
        return {key: _translate_value(item, language) for key, item in value.items()}
    return value


def translate_recipe(recipe: dict[str, Any], language: str) -> dict[str, Any]:
    """Translate recipe payload values while preserving structure."""
    return _translate_value(recipe, language)


def translate_recipe_to_hindi(recipe: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible helper for Hindi translation."""
    return translate_recipe(recipe, "hi")


if __name__ == "__main__":
    sample = "How to cook rice"
    print("Input:", sample)
    print("Hindi:", translate_text(sample, "hi"))
    print("Tamil:", translate_text(sample, "ta"))
    print("Telugu:", translate_text(sample, "te"))
    print("Kannada:", translate_text(sample, "kn"))
