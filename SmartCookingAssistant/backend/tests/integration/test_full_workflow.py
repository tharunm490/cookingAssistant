from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from backend import main


@pytest.mark.integration
def test_full_flow_login_to_voice(monkeypatch: pytest.MonkeyPatch, image_bytes: bytes, sample_recipe_payload: dict) -> None:
    if os.getenv("RUN_DB_TESTS", "0") != "1":
        pytest.skip("Integration tests disabled. Set RUN_DB_TESTS=1 to enable.")

    required = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    if any(not os.getenv(key) for key in required):
        pytest.skip("Integration DB env vars are not set.")

    class FakeDetector:
        def detect_from_images(self, images):
            return [{"ingredient": "tomato", "confidence": 0.99}, {"ingredient": "rice", "confidence": 0.95}]

    monkeypatch.setattr(main, "_get_or_create_detector", lambda: FakeDetector())
    monkeypatch.setattr(main.recipe_generator, "generate", lambda ingredients, preferences: sample_recipe_payload)
    monkeypatch.setattr(main.tts_service, "create_audio", lambda recipe, language: {"audio_path": "flow.mp3", "audio_url": "/audio/flow.mp3"})

    client = TestClient(main.app)

    signup = client.post("/signup", json={"name": "Flow QA", "email": "flowqa@example.com", "password": "flow123"})
    assert signup.status_code in {201, 409}

    login = client.post("/login", json={"email": "flowqa@example.com", "password": "flow123"})
    assert login.status_code == 200
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post("/upload-image", headers=headers, files={"file": ("flow.jpg", image_bytes, "image/jpeg")})
    assert upload.status_code == 201

    detect = client.post("/detect-ingredients", files={"images": ("flow.jpg", image_bytes, "image/jpeg")})
    assert detect.status_code == 200

    recipe_resp = client.post(
        "/generate-recipe",
        headers=headers,
        json={
            "ingredients": ["tomato", "rice"],
            "dish_name": "tomato bath",
            "health_goals": ["diabetic_friendly"],
            "language": "kn",
        },
    )
    assert recipe_resp.status_code == 200
    recipe_payload = recipe_resp.json()
    assert "extra_ingredients" in recipe_payload

    save = client.post(
        "/save-recipe",
        headers=headers,
        json={
            "recipe_name": recipe_payload["recipe_name"],
            "meal_type": "lunch",
            "diet_type": "veg",
            "language": "kn",
            "recipe_json": recipe_payload,
            "nutrition": recipe_payload.get("nutrition", {}),
        },
    )
    assert save.status_code == 201

    voice = client.post("/text-to-speech", json={"recipe": recipe_payload, "language": "kn"})
    assert voice.status_code == 200
    assert voice.json().get("audio_url")

    history = client.get("/my-recipes", headers=headers)
    assert history.status_code == 200
    assert isinstance(history.json().get("recipes", []), list)
