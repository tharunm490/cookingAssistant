from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from huggingface_hub import InferenceClient

from .config import settings
from .utils import extract_json_object, normalize_text_list

PROMPT_TEMPLATE = """Generate a complete cooking recipe.

User ingredients: {ingredients}
Preferences: {preferences}

Rules:

* Add missing essential ingredients
* Include measurements
* Provide detailed steps
* Adjust based on age group and health goal
* If dessert selected, generate sweet dish

Return STRICT JSON format:
{{
    "recipe_name": "string",
    "input_ingredients": ["string"],
    "extra_ingredients": ["string"],
    "ingredients_with_measurements": [
        {{"ingredient": "string", "measurement": "string"}}
    ],
    "steps": ["string"],
    "cooking_time": "Prep: X minutes | Total: Y minutes",
    "servings": "2",
    "nutrition": {{
        "calories": "string",
        "protein": "string",
        "carbs": "string",
        "fat": "string"
    }}
}}

Return JSON only. No markdown fences.
"""

CHEF_RULES_APPENDIX = """
You are a professional chef AI trained in Indian and global cooking. Generate realistic, accurate, and dish-specific recipes.

GLOBAL RULES:
1. Always identify a real dish name from ingredients. Never use generic names.
2. Always add missing essential ingredients where needed.
3. Use realistic measurements only.
4. Do not add irrelevant ingredients.
5. Do not repeat ingredients.

DISH IDENTIFICATION EXAMPLES:
- carrot + milk => Carrot Halwa
- rice + milk => Kheer
- tomato + onion => Curry
- flour + sugar => Dessert/Bakery

CATEGORY RULES:
- Curry/Gravy: include oil or ghee, onion, tomato, core spices; saute base, add spices, simmer.
- Dry/Stir-fry: include oil and basic spices; no gravy.
- Dessert: include sugar or jaggery and milk or ghee; optional cardamom/nuts; avoid salt and avoid oil.
- Rice dish: include rice and water; wash rice and cook properly.
- Breakfast: keep quick and simple.
- Soup: include water or stock; boil then simmer.
- Non-veg: include safe and sufficient cook time.
- Snacks: quick and simple.

STEPS:
- Numbered, specific, ingredient-matching, beginner-friendly.

VALIDATION:
- Ensure ingredients are logical.
- Ensure quantities are realistic.
- Ensure steps match ingredients.
"""

ALLOWED_LANGUAGES = {"en", "hi", "ta", "te", "kn"}


@dataclass(frozen=True)
class RecipePreferences:
    meal_type: str = "dinner"
    diet: str = "veg"
    spice_level: str = "medium"
    age_group: str = "adults"
    health_goals: list[str] = field(default_factory=list)
    servings: int = 2
    language: str = "en"
    user_text: str = ""


class RecipeGenerator:
    def __init__(self) -> None:
        self.client = InferenceClient(model=settings.recipe_model_name, token=settings.hf_token)

    def build_prompt(self, ingredients: list[str], preferences: RecipePreferences) -> str:
        clean_ingredients = normalize_text_list(ingredients)
        pref_blob = {
            "meal_type": preferences.meal_type,
            "diet": preferences.diet,
            "spice_level": preferences.spice_level,
            "age_group": preferences.age_group,
            "health_goals": preferences.health_goals,
            "servings": preferences.servings,
            "language": preferences.language,
            "user_text": preferences.user_text,
        }
        return (PROMPT_TEMPLATE.format(
            ingredients=", ".join(clean_ingredients) if clean_ingredients else "none",
            preferences=pref_blob,
        ) + "\n" + CHEF_RULES_APPENDIX)

    def generate(self, ingredients: list[str], preferences: RecipePreferences) -> dict[str, Any]:
        clean_ingredients = normalize_text_list(ingredients)
        auto_extras = self._get_missing_essentials(clean_ingredients, preferences)
        prompt_ingredients = normalize_text_list(clean_ingredients + auto_extras)

        if settings.hf_token:
            prompt = self.build_prompt(prompt_ingredients, preferences)
            try:
                response = self.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=800,
                )

                payload_text = self._extract_chat_content(response)
                if payload_text:
                    return self._normalize_recipe_payload(payload_text, clean_ingredients, auto_extras, preferences)
            except Exception:
                pass

        return self._fallback_recipe(clean_ingredients, auto_extras, preferences)

    def _extract_chat_content(self, response: Any) -> str:
        try:
            choices = getattr(response, "choices", None)
            if choices and len(choices) > 0:
                message = getattr(choices[0], "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                    if isinstance(content, str):
                        return content
                    if isinstance(message, dict):
                        msg_content = message.get("content")
                        if isinstance(msg_content, str):
                            return msg_content
        except Exception:
            pass

        if isinstance(response, str):
            return response
        return ""

    def _normalize_recipe_payload(
        self,
        payload: str,
        ingredients: list[str],
        forced_extras: list[str],
        preferences: RecipePreferences,
    ) -> dict[str, Any]:
        try:
            parsed = extract_json_object(payload)
            return self._post_process(parsed, ingredients, forced_extras, preferences)
        except Exception:
            return self._fallback_recipe(ingredients, forced_extras, preferences)

    def _post_process(
        self,
        recipe: dict[str, Any],
        ingredients: list[str],
        forced_extras: list[str],
        preferences: RecipePreferences,
    ) -> dict[str, Any]:
        input_ingredients = normalize_text_list(recipe.get("input_ingredients", []))
        if not input_ingredients:
            input_ingredients = self._extract_ingredient_names(recipe.get("ingredients"), ingredients)

        extra_ingredients = normalize_text_list(recipe.get("extra_ingredients", []))
        extra_ingredients = normalize_text_list(extra_ingredients + forced_extras)

        if not input_ingredients:
            input_ingredients = ingredients or ["tomato", "onion"]

        model_rows = recipe.get("ingredients_with_measurements", [])
        if not model_rows:
            model_rows = self._coerce_ingredient_rows(recipe.get("ingredients"))

        rows = self._normalize_ingredient_rows(
            model_rows,
            input_ingredients,
            extra_ingredients,
        )

        model_steps = recipe.get("steps", [])
        if not model_steps:
            model_steps = recipe.get("instructions", [])

        numbered_steps = self._normalize_steps(model_steps, preferences)
        nutrition = self._normalize_nutrition(self._extract_nutrition_payload(recipe), rows, preferences.servings)

        return {
            "recipe_name": str(recipe.get("recipe_name") or self._fallback_name(input_ingredients, preferences)),
            "input_ingredients": input_ingredients,
            "extra_ingredients": extra_ingredients,
            "ingredients_with_measurements": rows,
            "steps": numbered_steps,
            "cooking_time": str(recipe.get("cooking_time") or self._estimate_cooking_time(preferences, rows)),
            "servings": str(recipe.get("servings") or preferences.servings),
            "meal_type": preferences.meal_type,
            "nutrition": nutrition,
        }

    def _extract_ingredient_names(self, model_ingredients: Any, fallback: list[str]) -> list[str]:
        if not isinstance(model_ingredients, list):
            return normalize_text_list(fallback)

        names: list[str] = []
        for item in model_ingredients:
            if isinstance(item, dict):
                name = str(item.get("ingredient") or item.get("name") or "").strip().lower()
                if name:
                    names.append(name)
            elif isinstance(item, str):
                name = item.strip().lower()
                if name:
                    names.append(name)

        merged = names if names else fallback
        return normalize_text_list(merged)

    def _coerce_ingredient_rows(self, model_ingredients: Any) -> list[dict[str, str]]:
        if not isinstance(model_ingredients, list):
            return []

        rows: list[dict[str, str]] = []
        for item in model_ingredients:
            if isinstance(item, dict):
                ingredient = str(item.get("ingredient") or item.get("name") or "").strip().lower()
                measurement = str(item.get("measurement") or "").strip()
                quantity = str(item.get("quantity") or "").strip()
                unit = str(item.get("unit") or "").strip()

                if not measurement and quantity:
                    measurement = f"{quantity} {unit}".strip()

                if ingredient:
                    rows.append({"ingredient": ingredient, "measurement": measurement})
            elif isinstance(item, str):
                ingredient = item.strip().lower()
                if ingredient:
                    rows.append({"ingredient": ingredient, "measurement": ""})

        return rows

    def _extract_nutrition_payload(self, recipe: dict[str, Any]) -> Any:
        nutrition = recipe.get("nutrition")
        if isinstance(nutrition, dict):
            return nutrition

        top_level = {
            "calories": recipe.get("calories"),
            "protein": recipe.get("protein"),
            "carbs": recipe.get("carbs"),
            "fat": recipe.get("fat"),
        }
        if any(value for value in top_level.values()):
            return top_level
        return {}

    def _fallback_name(self, ingredients: list[str], preferences: RecipePreferences) -> str:
        dish = self._infer_dish_name(ingredients, preferences)
        return dish

    def _infer_dish_name(self, ingredients: list[str], preferences: RecipePreferences) -> str:
        ing = set(normalize_text_list(ingredients))
        meal = preferences.meal_type.lower()

        if {"carrot", "milk"}.issubset(ing):
            return "Carrot Halwa"
        if {"rice", "milk"}.issubset(ing):
            return "Rice Kheer"
        if {"tomato", "onion"}.issubset(ing):
            return "Onion Tomato Curry"
        if {"flour", "sugar"}.issubset(ing) or {"maida", "sugar"}.issubset(ing):
            return "Sweet Flour Pancakes"
        if meal == "dessert":
            if "semolina" in ing:
                return "Rava Kesari"
            if "milk" in ing:
                return "Milk Pudding"
            return "Jaggery Coconut Ladoo"
        if meal == "breakfast":
            if "egg" in ing:
                return "Masala Omelette"
            if "semolina" in ing:
                return "Vegetable Upma"
            return "Savory Breakfast Scramble"
        if "rice" in ing:
            return "Vegetable Rice"
        if any(x in ing for x in ["chicken", "fish", "mutton", "prawns", "crab"]):
            return "Spiced Protein Curry"
        if meal == "snack":
            return "Crispy Veg Snack"
        return "Home Style Mixed Vegetable Curry"

    def _measurement_for(self, ingredient: str) -> str:
        liquid_words = {"milk", "water", "curd", "yogurt", "cream", "buttermilk", "oil"}
        spice_words = {"salt", "turmeric", "cardamom", "garam masala", "black pepper", "cinnamon"}
        nut_words = {"nuts", "almonds", "cashew", "cashews"}

        if ingredient in liquid_words:
            return "1 cup"
        if ingredient in spice_words:
            return "1/2 tsp"
        if ingredient in nut_words:
            return "2 tbsp"
        if "powder" in ingredient:
            return "1 tsp"
        if ingredient in {"sugar", "jaggery"}:
            return "3 tbsp"
        if ingredient in {"ghee", "butter"}:
            return "1 tbsp"
        if ingredient in {"onion", "tomato", "potato", "carrot"}:
            return "1 cup chopped"
        if ingredient in {"rice", "flour", "maida", "atta", "semolina", "oats", "quinoa"}:
            return "1 cup"
        if ingredient in {"chicken", "fish", "mutton", "paneer"}:
            return "250 grams"
        if ingredient in {"egg", "boiled egg", "fried egg"}:
            return "2 nos"
        return "2 cups"

    def _normalize_ingredient_rows(
        self,
        rows: Any,
        input_ingredients: list[str],
        extra_ingredients: list[str],
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        if isinstance(rows, list):
            for item in rows:
                ingredient = ""
                measurement = ""
                if isinstance(item, dict):
                    ingredient = str(item.get("ingredient", "")).strip().lower()
                    measurement = str(item.get("measurement", "")).strip()
                elif isinstance(item, str):
                    ingredient = item.strip().lower()
                if ingredient:
                    normalized.append(
                        {
                            "ingredient": ingredient,
                            "measurement": measurement or self._measurement_for(ingredient),
                        }
                    )

        required = normalize_text_list(input_ingredients + extra_ingredients)
        existing = {r["ingredient"] for r in normalized}
        for ingredient in required:
            if ingredient not in existing:
                normalized.append({"ingredient": ingredient, "measurement": self._measurement_for(ingredient)})

        return normalized

    def _normalize_steps(self, steps: Any, preferences: RecipePreferences) -> list[str]:
        if isinstance(steps, list):
            clean_steps = [str(step).strip() for step in steps if str(step).strip()]
        else:
            clean_steps = []

        if not clean_steps:
            clean_steps = self._fallback_steps(preferences)

        numbered = []
        for index, step in enumerate(clean_steps, start=1):
            normalized_step = step
            if not step[:3].strip().startswith(str(index)):
                normalized_step = f"{index}. {step}"
            numbered.append(normalized_step)
        return numbered

    def _fallback_steps(self, preferences: RecipePreferences) -> list[str]:
        meal = preferences.meal_type.lower()
        age = preferences.age_group.lower()

        if meal == "dessert":
            return [
                "Heat 1 tbsp ghee in a heavy pan on low flame.",
                "Add the main sweet ingredient and saute for 4 to 5 minutes until aromatic.",
                "Pour in milk and cook on medium flame until the mixture thickens.",
                "Add sugar or jaggery and stir continuously until glossy.",
                "Mix in cardamom and chopped nuts, then cook for 1 more minute.",
                "Serve warm as dessert.",
            ]

        if meal in {"lunch", "dinner"}:
            spice_note = "Keep spices mild for easy digestion." if age == "elderly" else "Adjust chilli based on spice preference."
            return [
                "Heat 1 to 2 tbsp oil or ghee in a pan.",
                "Add chopped onion and saute until translucent.",
                "Add chopped tomato and cook until soft and pulpy.",
                "Add turmeric, chilli powder, and other spices; saute for 30 seconds.",
                "Add main ingredients with a little water and cook until tender.",
                f"Simmer for 5 minutes so flavors combine. {spice_note}",
                "Finish with herbs and serve hot.",
            ]

        if meal == "breakfast":
            return [
                "Prep all ingredients and keep them ready.",
                "Heat 1 tbsp oil or ghee in a pan.",
                "Cook the base ingredients for 2 to 3 minutes.",
                "Add the main breakfast ingredient and cook until done.",
                "Season lightly and serve immediately.",
            ]

        return [
            "Prepare all ingredients before starting.",
            "Heat 1 tbsp oil in a pan and add aromatics.",
            "Add main ingredients and cook until done.",
            "Adjust seasoning and serve warm.",
        ]

    def _normalize_nutrition(self, nutrition: Any, rows: list[dict[str, str]], servings: int) -> dict[str, str]:
        if isinstance(nutrition, dict):
            calories = str(nutrition.get("calories", ""))
            protein = str(nutrition.get("protein", ""))
            carbs = str(nutrition.get("carbs", ""))
            fat = str(nutrition.get("fat", ""))
            if calories and protein and carbs and fat:
                return {
                    "calories": calories,
                    "protein": protein,
                    "carbs": carbs,
                    "fat": fat,
                }

        ingredient_count = max(len(rows), 1)
        base_cal = 120 * ingredient_count
        base_protein = 4 * ingredient_count
        base_carbs = 8 * ingredient_count
        base_fat = 3 * ingredient_count
        servings = max(1, servings)

        return {
            "calories": f"{int(base_cal / servings)} kcal",
            "protein": f"{int(base_protein / servings)} g",
            "carbs": f"{int(base_carbs / servings)} g",
            "fat": f"{int(base_fat / servings)} g",
        }

    def _estimate_cooking_time(self, preferences: RecipePreferences, rows: list[dict[str, str]]) -> str:
        base = 20
        if preferences.meal_type == "dessert":
            base = 30
        if len(rows) > 8:
            base += 10
        return f"Prep: 10 minutes | Total: {base} minutes"

    def _get_missing_essentials(self, ingredients: list[str], preferences: RecipePreferences) -> list[str]:
        ing = set(ingredients)
        extras: list[str] = []

        meal = preferences.meal_type.lower()

        if meal == "dessert":
            for item in ["sugar", "milk", "ghee"]:
                if item not in ing:
                    extras.append(item)
            for opt in ["cardamom", "nuts"]:
                if opt not in ing:
                    extras.append(opt)
        else:
            for item in ["salt", "oil"]:
                if item not in ing:
                    extras.append(item)

            has_curry_like = any(x in ing for x in ["tomato", "onion", "chicken", "fish", "mutton", "paneer", "potato"])
            if has_curry_like:
                for item in ["onion", "tomato", "turmeric", "red chilli powder"]:
                    if item not in ing:
                        extras.append(item)

            if "rice" in ing:
                if "water" not in ing:
                    extras.append("water")

        if "carrot" in ing and "milk" in ing:
            for item in ["sugar", "ghee", "cardamom", "nuts"]:
                if item not in ing:
                    extras.append(item)

        if preferences.age_group == "kids" and preferences.spice_level == "high":
            # Keep kid recipes mild.
            extras.append("butter")
        if preferences.age_group == "elderly":
            for item in ["ginger", "soft vegetables"]:
                if item not in ing:
                    extras.append(item)

        goals = {goal.lower() for goal in preferences.health_goals}
        if "diabetic friendly" in goals:
            extras = [x for x in extras if x not in {"sugar"}]
            if "jaggery" in extras:
                extras.remove("jaggery")

        if "weight loss" in goals:
            extras = [x for x in extras if x != "ghee"]
        if "high protein" in goals:
            for item in ["paneer", "lentils"]:
                if item not in ing and item not in extras:
                    extras.append(item)

        return normalize_text_list(extras)

    def _fallback_recipe(
        self,
        ingredients: list[str],
        forced_extras: list[str],
        preferences: RecipePreferences,
    ) -> dict[str, Any]:
        input_ingredients = ingredients or ["tomato", "onion"]
        input_ingredients = normalize_text_list(input_ingredients)
        extra_ingredients = normalize_text_list(forced_extras)

        rows = self._normalize_ingredient_rows([], input_ingredients, extra_ingredients)
        steps = self._normalize_steps([], preferences)

        return {
            "recipe_name": self._fallback_name(input_ingredients, preferences),
            "input_ingredients": input_ingredients,
            "extra_ingredients": extra_ingredients,
            "ingredients_with_measurements": rows,
            "steps": steps,
            "cooking_time": self._estimate_cooking_time(preferences, rows),
            "servings": str(preferences.servings),
            "meal_type": preferences.meal_type,
            "nutrition": self._normalize_nutrition({}, rows, preferences.servings),
        }


def parse_text_hint_to_ingredients(text: str, known_ingredients: list[str]) -> list[str]:
    clean_text = text.lower().strip()
    if not clean_text:
        return []
    detected = [item for item in known_ingredients if item in clean_text]
    return normalize_text_list(detected)
