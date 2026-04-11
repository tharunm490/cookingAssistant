from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any
import shutil

import bcrypt
from jose import JWTError, jwt
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from io import BytesIO

from .clip_model import get_detector, ingredients_data
from .config import PROJECT_DIR, settings
from .db import get_db_connection
from .recipe_generator import RecipeGenerator, RecipePreferences, parse_text_hint_to_ingredients
from .tts_service import TTSService
from .translation_service import translate_recipe
from .utils import normalize_text_list

app = FastAPI(title="Smart Cooking Assistant", version="1.0.0")

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

_uploads_root = os.getenv("LOCALAPPDATA")
if _uploads_root:
    UPLOADS_DIR = Path(_uploads_root) / "SmartCookingAssistant" / "uploaded_images"
else:
    UPLOADS_DIR = PROJECT_DIR.parent / "SmartCookingAssistantUploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(settings.audio_route, StaticFiles(directory=settings.audio_dir), name="audio")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


recipe_generator = RecipeGenerator()
tts_service = TTSService()


_detector = None
_detector_lock = threading.Lock()


def _get_or_create_detector():
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = get_detector()
    return _detector


@app.on_event("startup")
def warm_detector() -> None:
    # Warm the detector in the background so startup does not block reload/server bind.
    threading.Thread(target=_get_or_create_detector, daemon=True).start()


class RecipeRequest(BaseModel):
    ingredients: list[str] = Field(default_factory=list)
    dish_name: str = ""
    meal_type: str = "dinner"
    diet: str = "veg"
    spice_level: str = "medium"
    age_group: str = "adults"
    health_goals: list[str] = Field(default_factory=list)
    servings: int = 2
    language: str = "en"
    user_text: str = ""


class TTSRequest(BaseModel):
    recipe: dict[str, Any]
    language: str = "en"


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class SaveRecipeRequest(BaseModel):
    recipe_name: str
    meal_type: str
    diet_type: str
    language: str
    recipe_json: dict[str, Any]
    nutrition: dict[str, Any] = Field(default_factory=dict)


class StoreIngredientsRequest(BaseModel):
    ingredients: list[str] = Field(default_factory=list)


class UpdatePasswordRequest(BaseModel):
    current_password: str
    new_password: str


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def _to_dict_row(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    return {}


def _normalize_health_goals(goals: list[str]) -> list[str]:
    alias_map = {
        "weight_loss": "weight loss",
        "high_protein": "high protein",
        "diabetic_friendly": "diabetic friendly",
        "low_fat": "low fat",
        "heart_healthy": "heart healthy",
    }

    normalized: list[str] = []
    seen: set[str] = set()
    for goal in goals:
        raw = str(goal).strip().lower().replace("-", "_").replace(" ", "_")
        mapped = alias_map.get(raw, str(goal).strip().lower().replace("_", " "))
        if mapped and mapped not in seen:
            seen.add(mapped)
            normalized.append(mapped)
    return normalized


def _infer_dish_name_from_text(text: str) -> str:
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return ""

    patterns = [
        r"(?:make|cook|prepare)\s+(?:a|an|the)?\s*([a-z\s-]{3,})",
        r"(?:want|need)\s+(?:to\s+)?(?:make|cook|prepare)\s+(?:a|an|the)?\s*([a-z\s-]{3,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            extracted = match.group(1).strip(" .,!?")
            extracted = re.split(r"\b(from|with|using|for|please|now|today)\b", extracted, maxsplit=1)[0].strip(" .,!?")
            extracted = re.sub(r"\s+", " ", extracted)
            return extracted

    # Fallback for direct phrases like "spinach dosa" or "rava idli".
    fallback = re.sub(r"[^a-z0-9\s-]", " ", cleaned)
    fallback = re.sub(r"\s+", " ", fallback).strip(" .,!?")
    return fallback


def get_current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization token.")

    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload.")
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token.") from exc

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT user_id, name, email FROM users WHERE user_id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="User does not exist.")
        return _to_dict_row(user)
    finally:
        cursor.close()
        conn.close()


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "message": "Smart Cooking Assistant API is running."}


@app.get("/me")
def get_me(user: dict[str, Any] = Depends(get_current_user)) -> JSONResponse:
    return JSONResponse({"user": user})


@app.post("/signup")
def signup(request: SignupRequest) -> JSONResponse:
    name = request.name.strip()
    email = request.email.strip().lower()
    password = request.password.strip()

    if not name or not email or not password:
        raise HTTPException(status_code=400, detail="Name, email, and password are required.")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            raise HTTPException(status_code=409, detail="Email already exists.")

        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
            (name, email, password_hash),
        )
        conn.commit()
        return JSONResponse({"message": "Signup successful."}, status_code=201)
    finally:
        cursor.close()
        conn.close()


@app.post("/login")
def login(request: LoginRequest, raw_request: Request) -> JSONResponse:
    email = request.email.strip().lower()
    password = request.password.strip()

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT user_id, name, email, password_hash FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        user = _to_dict_row(user)
        if not verify_password(password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        cursor.execute(
            "UPDATE users SET last_login = NOW(), login_count = login_count + 1 WHERE user_id = %s",
            (user["user_id"],),
        )

        ip_address = raw_request.client.host if raw_request.client else "unknown"
        device_info = raw_request.headers.get("user-agent", "unknown")[:255]
        cursor.execute(
            "INSERT INTO login_history (user_id, ip_address, device_info) VALUES (%s, %s, %s)",
            (user["user_id"], ip_address, device_info),
        )
        conn.commit()

        token = create_access_token({"user_id": user["user_id"], "email": user["email"]})
        return JSONResponse(
            {
                "access_token": token,
                "token_type": "bearer",
                "user": {
                    "user_id": user["user_id"],
                    "name": user["name"],
                    "email": user["email"],
                },
            }
        )
    finally:
        cursor.close()
        conn.close()


@app.post("/detect-ingredients")
async def detect_ingredients(images: list[UploadFile] = File(...)) -> JSONResponse:
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    pil_images: list[Image.Image] = []
    for upload in images:
        content = await upload.read()
        try:
            pil_images.append(Image.open(BytesIO(content)))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {upload.filename}") from exc

    detector = await run_in_threadpool(_get_or_create_detector)
    detected = await run_in_threadpool(detector.detect_from_images, pil_images)
    return JSONResponse(
        {
            "ingredients": detected,
            "top_ingredients": [item["ingredient"] for item in detected],
            "image_count": len(pil_images),
        }
    )


@app.post("/generate-recipe")
def generate_recipe(request: RecipeRequest, user: dict[str, Any] = Depends(get_current_user)) -> JSONResponse:
    flattened_ingredients: list[str] = []
    for group in ingredients_data.values():
        flattened_ingredients.extend(group)

    text_ingredients = parse_text_hint_to_ingredients(request.user_text, flattened_ingredients)
    all_ingredients = request.ingredients + text_ingredients

    if not all_ingredients and not request.user_text.strip():
        raise HTTPException(status_code=400, detail="Provide ingredients from image/text/voice before generating recipe.")

    normalized_health_goals = _normalize_health_goals(request.health_goals)
    dish_name = request.dish_name.strip() or _infer_dish_name_from_text(request.user_text)

    preferences = RecipePreferences(
        meal_type=request.meal_type,
        diet=request.diet,
        spice_level=request.spice_level,
        age_group=request.age_group,
        health_goals=normalized_health_goals,
        servings=max(1, request.servings),
        language=request.language,
        user_text=request.user_text,
        dish_name=dish_name,
    )

    # Keep the dependency explicit for route-level auth enforcement.
    _ = user
    recipe = recipe_generator.generate(all_ingredients, preferences)

    recipe = translate_recipe(recipe, request.language)

    return JSONResponse(recipe)


@app.post("/text-to-speech")
def text_to_speech(request: TTSRequest) -> JSONResponse:
    if not request.recipe:
        raise HTTPException(status_code=400, detail="Recipe payload is required.")

    try:
        audio_data = tts_service.create_audio(request.recipe, request.language)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JSONResponse(audio_data)


@app.post("/save-recipe")
def save_recipe(request: SaveRecipeRequest, user: dict[str, Any] = Depends(get_current_user)) -> JSONResponse:
    nutrition = request.nutrition or request.recipe_json.get("nutrition", {})

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO recipes
            (user_id, recipe_name, meal_type, diet_type, language, recipe_json, calories, protein, carbs, fat)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user["user_id"],
                request.recipe_name,
                request.meal_type,
                request.diet_type,
                request.language,
                json.dumps(request.recipe_json),
                str(nutrition.get("calories", "")),
                str(nutrition.get("protein", "")),
                str(nutrition.get("carbs", "")),
                str(nutrition.get("fat", "")),
            ),
        )
        conn.commit()
        return JSONResponse({"message": "Recipe saved successfully."}, status_code=201)
    finally:
        cursor.close()
        conn.close()


@app.post("/store-ingredients")
def store_ingredients(request: StoreIngredientsRequest, user: dict[str, Any] = Depends(get_current_user)) -> JSONResponse:
    ingredients = normalize_text_list(request.ingredients)
    if not ingredients:
        raise HTTPException(status_code=400, detail="At least one ingredient is required.")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO detected_ingredients (user_id, ingredients_json) VALUES (%s, %s)",
            (user["user_id"], json.dumps(ingredients)),
        )
        conn.commit()
        return JSONResponse({"message": "Ingredients stored successfully."}, status_code=201)
    finally:
        cursor.close()
        conn.close()


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), user: dict[str, Any] = Depends(get_current_user)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Image file is required.")

    suffix = Path(file.filename).suffix or ".jpg"
    safe_name = f"{uuid4().hex}{suffix}"
    destination = UPLOADS_DIR / safe_name

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO uploaded_images (user_id, image_path) VALUES (%s, %s)",
            (user["user_id"], str(destination)),
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return JSONResponse(
        {
            "message": "Image uploaded successfully.",
            "image_path": str(destination),
            "image_url": f"/uploads/{safe_name}",
        },
        status_code=201,
    )


@app.get("/my-recipes")
def my_recipes(user: dict[str, Any] = Depends(get_current_user)) -> JSONResponse:
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT recipe_id, recipe_name, meal_type, diet_type, language, recipe_json,
                   calories, protein, carbs, fat, created_at
            FROM recipes
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user["user_id"],),
        )
        rows = cursor.fetchall() or []

        recipes: list[dict[str, Any]] = []
        for row in rows:
            row_data = _to_dict_row(row)
            recipe_json_value = row_data.get("recipe_json")
            if isinstance(recipe_json_value, str):
                try:
                    recipe_payload = json.loads(recipe_json_value)
                except json.JSONDecodeError:
                    recipe_payload = {}
            elif isinstance(recipe_json_value, dict):
                recipe_payload = recipe_json_value
            else:
                recipe_payload = {}

            recipes.append(
                {
                    "recipe_id": row_data.get("recipe_id"),
                    "recipe_name": row_data.get("recipe_name"),
                    "meal_type": row_data.get("meal_type"),
                    "diet_type": row_data.get("diet_type"),
                    "language": row_data.get("language"),
                    "recipe_json": recipe_payload,
                    "nutrition": {
                        "calories": row_data.get("calories", ""),
                        "protein": row_data.get("protein", ""),
                        "carbs": row_data.get("carbs", ""),
                        "fat": row_data.get("fat", ""),
                    },
                    "created_at": str(row_data.get("created_at", "")),
                }
            )

        return JSONResponse({"recipes": recipes})
    finally:
        cursor.close()
        conn.close()


@app.put("/update-password")
def update_password(request: UpdatePasswordRequest, user: dict[str, Any] = Depends(get_current_user)) -> JSONResponse:
    current_password = request.current_password.strip()
    new_password = request.new_password.strip()

    if not current_password or not new_password:
        raise HTTPException(status_code=400, detail="Current and new password are required.")
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters.")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT password_hash FROM users WHERE user_id = %s", (user["user_id"],))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found.")

        row_data = _to_dict_row(row)
        if not verify_password(current_password, row_data["password_hash"]):
            raise HTTPException(status_code=401, detail="Current password is incorrect.")

        new_hash = hash_password(new_password)
        cursor.execute("UPDATE users SET password_hash = %s WHERE user_id = %s", (new_hash, user["user_id"]))
        conn.commit()
        return JSONResponse({"message": "Password updated successfully."})
    finally:
        cursor.close()
        conn.close()
