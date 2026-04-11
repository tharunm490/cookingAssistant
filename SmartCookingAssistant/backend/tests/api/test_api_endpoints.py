from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend import main


@dataclass
class _Store:
    users: list[dict[str, Any]] = field(default_factory=list)
    login_history: list[dict[str, Any]] = field(default_factory=list)
    recipes: list[dict[str, Any]] = field(default_factory=list)


class _Cursor:
    def __init__(self, store: _Store, dictionary: bool = False) -> None:
        self.store = store
        self.dictionary = dictionary
        self._result = None
        self._results = None

    def execute(self, query: str, params=None) -> None:
        sql = " ".join(query.lower().split())
        params = params or ()

        if "select user_id from users where email" in sql:
            email = params[0]
            user = next((u for u in self.store.users if u["email"] == email), None)
            self._result = {"user_id": user["user_id"]} if user else None
        elif "insert into users" in sql:
            uid = len(self.store.users) + 1
            self.store.users.append(
                {
                    "user_id": uid,
                    "name": params[0],
                    "email": params[1],
                    "password_hash": params[2],
                    "login_count": 0,
                }
            )
        elif "select user_id, name, email, password_hash from users where email" in sql:
            email = params[0]
            self._result = next((u for u in self.store.users if u["email"] == email), None)
        elif "update users set last_login" in sql:
            uid = params[0]
            user = next((u for u in self.store.users if u["user_id"] == uid), None)
            if user:
                user["login_count"] += 1
        elif "insert into login_history" in sql:
            self.store.login_history.append({"user_id": params[0], "ip": params[1], "device": params[2]})
        elif "insert into recipes" in sql:
            self.store.recipes.append({"user_id": params[0], "recipe_name": params[1]})
        elif "select recipe_id, recipe_name" in sql and "from recipes" in sql:
            uid = params[0]
            rows = [
                {
                    "recipe_id": i + 1,
                    "recipe_name": r["recipe_name"],
                    "meal_type": "lunch",
                    "diet_type": "veg",
                    "language": "en",
                    "recipe_json": "{}",
                    "calories": "100",
                    "protein": "3",
                    "carbs": "15",
                    "fat": "3",
                    "created_at": "2026-01-01 00:00:00",
                }
                for i, r in enumerate(self.store.recipes)
                if r["user_id"] == uid
            ]
            self._results = rows
        elif "insert into uploaded_images" in sql:
            return
        elif "select user_id, name, email from users where user_id" in sql:
            uid = params[0]
            user = next((u for u in self.store.users if u["user_id"] == uid), None)
            self._result = {"user_id": user["user_id"], "name": user["name"], "email": user["email"]} if user else None

    def fetchone(self):
        return self._result

    def fetchall(self):
        return self._results or []

    def close(self) -> None:
        return None


class _Conn:
    def __init__(self, store: _Store) -> None:
        self.store = store

    def cursor(self, dictionary: bool = False):
        return _Cursor(self.store, dictionary=dictionary)

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeDetector:
    def detect_from_images(self, images):
        return [{"ingredient": "tomato", "confidence": 0.99}]


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch, sample_recipe_payload: dict):
    store = _Store()
    monkeypatch.setattr(main, "get_db_connection", lambda: _Conn(store))
    monkeypatch.setattr(main, "_get_or_create_detector", lambda: _FakeDetector())
    monkeypatch.setattr(main.recipe_generator, "generate", lambda ingredients, preferences: sample_recipe_payload)
    monkeypatch.setattr(main.tts_service, "create_audio", lambda recipe, language: {"audio_path": "demo.mp3", "audio_url": "/audio/demo.mp3"})

    client = TestClient(main.app)
    return client


@pytest.mark.api
def test_signup_and_login(api_client: TestClient) -> None:
    signup = api_client.post("/signup", json={"name": "QA", "email": "qa@example.com", "password": "safe123"})
    assert signup.status_code == 201

    login = api_client.post("/login", json={"email": "qa@example.com", "password": "safe123"})
    assert login.status_code == 200
    data = login.json()
    assert "access_token" in data
    assert "user" in data


@pytest.mark.api
def test_detect_ingredients_endpoint(api_client: TestClient, image_bytes: bytes) -> None:
    response = api_client.post(
        "/detect-ingredients",
        files={"images": ("tomato.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "ingredients" in body
    assert body["ingredients"][0]["ingredient"] == "tomato"


@pytest.mark.api
def test_generate_recipe_and_voice_and_history(api_client: TestClient) -> None:
    signup = api_client.post("/signup", json={"name": "API", "email": "api@example.com", "password": "safe123"})
    assert signup.status_code == 201
    login = api_client.post("/login", json={"email": "api@example.com", "password": "safe123"})
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    recipe = api_client.post(
        "/generate-recipe",
        headers=headers,
        json={"ingredients": ["tomato", "rice"], "dish_name": "tomato bath", "language": "en"},
    )
    assert recipe.status_code == 200
    recipe_body = recipe.json()
    assert "recipe_name" in recipe_body
    assert "ingredients_with_measurements" in recipe_body

    save = api_client.post(
        "/save-recipe",
        headers=headers,
        json={
            "recipe_name": recipe_body["recipe_name"],
            "meal_type": "lunch",
            "diet_type": "veg",
            "language": "en",
            "recipe_json": recipe_body,
            "nutrition": recipe_body.get("nutrition", {}),
        },
    )
    assert save.status_code == 201

    voice = api_client.post("/text-to-speech", json={"recipe": recipe_body, "language": "en"})
    assert voice.status_code == 200
    assert "audio_url" in voice.json()

    history = api_client.get("/my-recipes", headers=headers)
    assert history.status_code == 200
    assert "recipes" in history.json()
