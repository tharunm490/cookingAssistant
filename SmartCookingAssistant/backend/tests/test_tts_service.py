from __future__ import annotations

from pathlib import Path

import pytest

from backend.tts_service import TTSService


@pytest.mark.unit
def test_voice_generation_creates_mp3(monkeypatch: pytest.MonkeyPatch, tmp_audio_dir: Path, sample_recipe_payload: dict) -> None:
    class FakeGTTS:
        def __init__(self, text: str, lang: str, slow: bool = False) -> None:
            self.text = text
            self.lang = lang
            self.slow = slow

        def save(self, path: str) -> None:
            Path(path).write_bytes(b"ID3fake")

    monkeypatch.setattr("backend.tts_service.gTTS", FakeGTTS)

    service = TTSService(output_dir=tmp_audio_dir)
    result = service.create_audio(sample_recipe_payload, language="en")

    audio_path = Path(result["audio_path"])
    assert audio_path.exists()
    assert audio_path.suffix == ".mp3"
    assert result["audio_url"].endswith(audio_path.name)
