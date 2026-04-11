from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from backend import main


class _FastDetector:
    def detect_from_images(self, images):
        return [{"ingredient": "tomato", "confidence": 0.97}]


@pytest.mark.performance
def test_image_detection_under_five_seconds(monkeypatch: pytest.MonkeyPatch, image_bytes: bytes) -> None:
    monkeypatch.setattr(main, "_get_or_create_detector", lambda: _FastDetector())
    client = TestClient(main.app)

    started = time.perf_counter()
    response = client.post(
        "/detect-ingredients",
        files={"images": ("tomato.jpg", image_bytes, "image/jpeg")},
    )
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 5.0


@pytest.mark.performance
def test_recipe_generation_under_eight_seconds(monkeypatch: pytest.MonkeyPatch, sample_recipe_payload: dict) -> None:
    main.app.dependency_overrides[main.get_current_user] = lambda: {"user_id": 1, "email": "qa@example.com"}
    monkeypatch.setattr(main.recipe_generator, "generate", lambda ingredients, preferences: sample_recipe_payload)

    client = TestClient(main.app)
    started = time.perf_counter()
    response = client.post(
        "/generate-recipe",
        json={"ingredients": ["tomato", "rice"], "dish_name": "tomato bath", "health_goals": ["diabetic_friendly"]},
        headers={"Authorization": "Bearer token"},
    )
    elapsed = time.perf_counter() - started

    main.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert elapsed < 8.0


@pytest.mark.performance
def test_voice_generation_under_five_seconds(monkeypatch: pytest.MonkeyPatch, sample_recipe_payload: dict) -> None:
    monkeypatch.setattr(main.tts_service, "create_audio", lambda recipe, language: {"audio_path": "x.mp3", "audio_url": "/audio/x.mp3"})
    client = TestClient(main.app)

    started = time.perf_counter()
    response = client.post("/text-to-speech", json={"recipe": sample_recipe_payload, "language": "en"})
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 5.0
