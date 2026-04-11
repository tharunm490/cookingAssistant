from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image


@pytest.fixture
def sample_recipe_payload() -> dict[str, Any]:
    return {
        "recipe_name": "Tomato Bath",
        "input_ingredients": ["tomato", "rice"],
        "matched_ingredients": ["tomato", "rice"],
        "missing_ingredients": ["salt", "oil"],
        "extra_ingredients": ["salt", "oil"],
        "ingredients_with_measurements": [
            {"ingredient": "tomato", "measurement": "2 medium"},
            {"ingredient": "rice", "measurement": "1 cup"},
            {"ingredient": "salt", "measurement": "to taste"},
            {"ingredient": "oil", "measurement": "1 tbsp"},
        ],
        "steps": ["1. Heat oil.", "2. Add tomato.", "3. Add rice and cook."],
        "cooking_time": "20 minutes",
        "servings": "2",
        "meal_type": "lunch",
        "nutrition": {
            "calories": "320 kcal",
            "protein": "7 g",
            "carbs": "45 g",
            "fat": "10 g",
        },
        "health_optimization": "diabetic friendly",
        "source": "dataset_exact",
        "engine_used": "csv",
        "generation_source": "dataset_exact",
    }


@pytest.fixture
def image_bytes() -> bytes:
    image = Image.new("RGB", (16, 16), color=(220, 40, 40))
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def tmp_audio_dir(tmp_path: Path) -> Path:
    path = tmp_path / "audio"
    path.mkdir(parents=True, exist_ok=True)
    return path
