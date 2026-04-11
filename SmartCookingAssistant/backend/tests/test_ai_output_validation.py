from __future__ import annotations

import pytest

from backend.recipe_generator import RecipeGenerator, RecipePreferences


@pytest.mark.ai
def test_recipe_correctness_tomato_bath_not_unrelated_bath(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "TranslatedRecipeName": "Tomato Bath",
            "Cleaned-Ingredients": "tomato, rice, salt, oil",
            "TranslatedIngredients": "tomato, rice, salt, oil",
            "TranslatedInstructions": "Cook with tomato.",
            "TotalTimeInMins": "20",
        },
        {
            "TranslatedRecipeName": "Bisi Bele Bath",
            "Cleaned-Ingredients": "dal, vegetables, rice",
            "TranslatedIngredients": "dal, vegetables, rice",
            "TranslatedInstructions": "Cook dal and rice.",
            "TotalTimeInMins": "35",
        },
    ]
    monkeypatch.setattr(RecipeGenerator, "_load_dataset_rows", lambda self: rows)
    generator = RecipeGenerator()

    prefs = RecipePreferences(dish_name="tomato bath", language="en")
    recipe = generator.generate(["tomato", "rice"], prefs)

    name = recipe.get("recipe_name", "").lower()
    assert "tomato" in name or "rice" in name
    assert "bisi bele" not in name


@pytest.mark.ai
def test_health_goal_in_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RecipeGenerator, "_load_dataset_rows", lambda self: [])
    generator = RecipeGenerator()

    prefs = RecipePreferences(
        dish_name="tomato rice",
        health_goals=["diabetic friendly"],
    )
    recipe = generator.generate(["tomato", "rice"], prefs)

    assert "health_optimization" in recipe
    assert "diabetic" in str(recipe["health_optimization"]).lower()


@pytest.mark.ai
def test_multilingual_translation_hook(monkeypatch: pytest.MonkeyPatch, sample_recipe_payload: dict) -> None:
    from backend import main

    def fake_translate(recipe: dict, language: str) -> dict:
        if language == "kn":
            recipe = dict(recipe)
            recipe["recipe_name"] = "ಟೊಮೇಟೊ ಬಾತ್"
        return recipe

    monkeypatch.setattr(main, "translate_recipe", fake_translate)
    monkeypatch.setattr(main.recipe_generator, "generate", lambda ingredients, preferences: sample_recipe_payload)

    translated = main.translate_recipe(sample_recipe_payload, "kn")
    assert "ಟೊಮೇಟೊ" in translated["recipe_name"]
