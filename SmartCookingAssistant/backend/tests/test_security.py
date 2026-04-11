from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from backend import main


@dataclass
class _FakeStore:
    users: list[dict] = field(default_factory=list)
    login_history: list[dict] = field(default_factory=list)


class _FakeCursor:
    def __init__(self, store: _FakeStore) -> None:
        self.store = store
        self._result = None

    def execute(self, query: str, params=None) -> None:
        sql = " ".join(query.lower().split())
        params = params or ()

        if "select user_id from users where email" in sql:
            email = params[0]
            user = next((u for u in self.store.users if u["email"] == email), None)
            self._result = {"user_id": user["user_id"]} if user else None
        elif "insert into users" in sql:
            user_id = len(self.store.users) + 1
            self.store.users.append({"user_id": user_id, "name": params[0], "email": params[1], "password_hash": params[2], "login_count": 0})
            self._result = None
        elif "select user_id, name, email, password_hash from users where email" in sql:
            email = params[0]
            self._result = next((u for u in self.store.users if u["email"] == email), None)
        elif "update users set last_login" in sql:
            uid = params[0]
            for u in self.store.users:
                if u["user_id"] == uid:
                    u["login_count"] += 1
        elif "insert into login_history" in sql:
            self.store.login_history.append({"user_id": params[0], "ip": params[1], "device": params[2]})
        else:
            self._result = None

    def fetchone(self):
        return self._result

    def close(self) -> None:
        return None


class _FakeConn:
    def __init__(self, store: _FakeStore) -> None:
        self.store = store

    def cursor(self, dictionary: bool = False):
        return _FakeCursor(self.store)

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.mark.security
def test_invalid_login_rejected_and_sql_injection_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _FakeStore()
    monkeypatch.setattr(main, "get_db_connection", lambda: _FakeConn(store))
    client = TestClient(main.app)

    signup = client.post("/signup", json={"name": "QA", "email": "qa@example.com", "password": "safe123"})
    assert signup.status_code == 201

    attack = client.post("/login", json={"email": "' OR 1=1 --", "password": "any"})
    assert attack.status_code == 401


@pytest.mark.security
def test_invalid_file_upload_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDetector:
        def detect_from_images(self, images):
            return []

    monkeypatch.setattr(main, "_get_or_create_detector", lambda: FakeDetector())
    client = TestClient(main.app)

    response = client.post(
        "/detect-ingredients",
        files={"images": ("bad.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 400


@pytest.mark.security
def test_password_hashed_on_signup(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _FakeStore()
    monkeypatch.setattr(main, "get_db_connection", lambda: _FakeConn(store))
    client = TestClient(main.app)

    raw_password = "plain-secret"
    response = client.post("/signup", json={"name": "QA", "email": "hash@example.com", "password": raw_password})

    assert response.status_code == 201
    assert store.users
    assert store.users[0]["password_hash"] != raw_password
    assert store.users[0]["password_hash"].startswith("$2")
