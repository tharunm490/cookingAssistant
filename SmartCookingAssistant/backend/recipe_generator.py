from __future__ import annotations

import csv
from dataclasses import dataclass, field
import re
from typing import Any

from huggingface_hub import InferenceClient

from .config import PROJECT_DIR, settings
from .utils import extract_json_object, normalize_text_list

HEALTH_GOALS_RULES = """
HEALTH GOALS ADAPTATION:
When user has selected specific health goals, apply these transformations:

1. WEIGHT LOSS:
   - Reduce oil/ghee usage (use minimal, ~1-2 tsp instead of 3-4 tsp)
   - Prefer grilling, steaming, or boiling over frying
   - Add high-fiber vegetables (spinach, broccoli, bell peppers)
   - Avoid added sugar; use natural sweetness where possible
   - Target calories: moderate portions, emphasize vegetables
   - Note in steps: "Light oil for weight-conscious cooking"

2. HIGH PROTEIN:
   - Add paneer, tofu, eggs, or legumes (dal, chickpeas)
   - Increase protein content in nutrition output
   - Include Greek yogurt or cottage cheese where applicable
   - Use dal/lentils as base for curry
   - Target protein: 20-30g per serving

3. DIABETIC FRIENDLY:
   - Completely avoid sugar and refined carbs
   - Use whole grains (brown rice, jowar, bajra) instead of white rice
   - Focus on low glycemic index vegetables
   - Include fiber-rich ingredients
   - Minimal salt (reduce by 30%)
   - No refined flour; use whole wheat or millet flour
   - Note: "Low GI, sugar-free, diabetes-friendly recipe"

4. LOW FAT:
   - Minimal oil (1 tsp maximum per serving)
   - Prefer steaming, boiling, or grilling
   - Use non-fat or low-fat dairy
   - Remove skin from chicken/meat
   - Avoid coconut milk and heavy cream
   - Fat target: <5-8g per serving

5. HEART HEALTHY:
   - Very minimal salt (reduce by 50%)
   - Use olive oil or olive-based cooking
   - Avoid trans fats and saturated fats (no ghee, butter minimally)
   - Include heart-healthy fats (nuts, seeds, olive oil sparingly)
   - Add heart-protective spices: turmeric, cinnamon
   - Prefer fish/lean meat/plant-based over red meat
   - High fiber content (oats, legumes, vegetables)
   - Note: "Heart-healthy, low sodium, balanced nutrition"

OUTPUT REQUIREMENT:
- For each health goal applied, add a note in the first step or as a prefix
- Example: "Heart Healthy Modification: Using olive oil instead of ghee..."
- Ensure nutrition output reflects the modifications
"""

PROMPT_TEMPLATE = """Generate a realistic Indian recipe.

User ingredients: {ingredients}
User requested dish: {requested_dish}
Preferences: {preferences}
User health goals: {health_goals}

Rules:

* Prioritize requested dish strictly when provided
* Never change dish category (example: dosa must remain dosa)
* Add missing essential ingredients
* Include realistic measurements
* Provide detailed steps
* Adjust based on age group and health goal
* Mention missing ingredients separately as extra_ingredients
* Recipe must follow authentic Indian culinary techniques where applicable

Return STRICT JSON format:
{{
    "recipe_name": "string",
    "input_ingredients": ["string"],
    "matched_ingredients": ["string"],
    "extra_ingredients": ["string"],
    "ingredients_with_measurements": [
        {{"ingredient": "string", "measurement": "string"}}
    ],
    "steps": ["string"],
    "time_breakdown": {{
        "active_prep": "X minutes",
        "passive_time": "X minutes",
        "cooking_time": "X minutes",
        "total_time": "X minutes"
    }},
    "cooking_time": "Active Prep: X minutes | Passive: X minutes | Cooking: X minutes | Total: X minutes",
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

STRICT_REALISM_RULES = """
STRICT REALISM RULES (MUST FOLLOW):
1. Use only realistic ingredient quantities.
2. Hard limits:
    - cumin seeds: max 1 tsp
    - mustard seeds: max 1 tsp
    - turmeric: max 1/2 tsp
    - chili powder/red chilli powder: max 1 tsp
    - water: 1 to 3 cups
    - milk: 1 to 2 cups
    - vegetables (onion/tomato/potato/carrot/etc): 1 to 3 medium pieces
    - oil: max 2 tbsp
    - salt: always write "to taste"
3. Never output unrealistic values (example: 2 cups cumin seeds, 1 cup turmeric).
4. Dish identification must be correct:
    - dal + tamarind => Sambar
    - carrot + milk => Carrot Halwa
5. Add missing essentials when required by dish logic.
6. Ensure no duplicate ingredients and no excessive measurements.
7. Priority: accuracy over creativity.
"""

CHEF_RULES_APPENDIX = """
You are an expert Indian Culinary Consultant with 20+ years of experience in North Indian (Mughlai, Punjabi, Awadhi) and South Indian (Tamil, Kerala, Andhra, Kannada) cuisines. Generate realistic, restaurant-grade, dish-specific recipes.

GLOBAL RULES:
1. Always identify a real dish name from ingredients. Never use generic names.
2. Always add missing essential ingredients where needed.
3. Use realistic measurements only.
4. Do not add irrelevant ingredients.
5. Do not repeat ingredients.
6. Always include a realistic time breakdown with active prep, passive time, stove cooking time, and total time.
7. Total time must include passive steps such as soaking, marination, fermentation, resting, and dum.
8. Use authentic professional ratios where relevant:
    - Dosa batter: 3:1 rice to urad dal.
    - Biryani: 1:1 meat to rice (by weight).
    - Standard tadka fat-to-whole-spice ratio: about 2 tbsp fat per 1 tsp whole spices.
9. Mention heat level in steps (low, medium, high, low dum, high saute) and add visual doneness cues.
10. For texture-sensitive vegetables or greens, specify texture control (puree, finely chopped, julienne, or garnish finish).

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

INDIAN TADKA OR VAGHAR ORDER (MANDATORY):
- North Indian gravies and masalas:
    1) Heat oil or ghee,
    2) Add whole spices if used,
    3) Add onion and cook until golden brown,
    4) Add ginger-garlic and saute until raw smell disappears,
    5) Add tomato and powdered spices,
    6) Cook until oil separates.
- South Indian tempering:
    1) Mustard seeds first (allow to crackle),
    2) urad dal next (light golden),
    3) curry leaves and chilies next,
    4) then onions or other aromatics as required.

STEPS:
- Numbered, specific, ingredient-matching, beginner-friendly.
- Include exact texture instruction where needed, for example: puree spinach smooth, keep onions finely chopped, keep vegetables bite-size.
- Include visual cues, for example: cook masala until oil separates; cook dosa until surface bubbles and edges lift.

VALIDATION:
- Ensure ingredients are logical.
- Ensure quantities are realistic.
- Ensure steps match ingredients.
- Ensure time_breakdown is internally consistent (active prep + passive + cooking = total).
"""

ALLOWED_LANGUAGES = {"en", "hi", "ta", "te", "kn"}

DISH_ALIAS_MAP: dict[str, list[str]] = {
    "tomato bath": ["tomato bath", "tomato rice"],
    "tomato rice": ["tomato rice", "tomato bath"],
    "spinach dosa": ["spinach dosa", "palak dosa"],
    "sambar": ["sambar"],
    "carrot halwa": ["carrot halwa", "gajar halwa"],
    "halwa": ["halwa"],
    "dosa": ["dosa"],
    "rice": ["rice", "pulao", "bath"],
}

DISH_TYPES = {"dosa", "rice", "sambar", "halwa", "pulao", "bath", "curry", "kheer", "upma"}
MAJOR_INGREDIENTS = {"tomato", "spinach", "palak", "carrot", "onion", "potato", "paneer", "chicken", "dal"}
GENERIC_QUERY_TOKENS = {
    "recipe", "dish", "food", "bath", "rice", "dosa", "sambar", "halwa",
    "i", "want", "make", "cook", "prepare", "how", "to", "please", "show", "me",
}
UNRELATED_VARIANT_KEYWORDS = {"bisi bele", "millet", "barnyard", "smoothie", "pulao"}
FILLER_WORDS = {"i", "want", "to", "make", "how", "can", "cook", "prepare"}
DISH_TYPE_EQUIVALENTS: dict[str, set[str]] = {
    "bath": {"bath", "rice"},
    "rice": {"rice", "bath", "pulao"},
}
# PHASE 4: Composite Recipe Detection
COMPOSITE_RECIPE_KEYWORDS = {"sizzler", "platter", "combo", "with", "and", "&", "sauce", "mix"}
DESCRIPTOR_KEYWORDS = {"recipe", "style", "way", "method", "traditional", "easy", "quick", "simple"}


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
    dish_name: str = ""


class RecipeGenerator:
    def __init__(self) -> None:
        self.client = InferenceClient(model=settings.recipe_model_name, token=settings.hf_token)
        self.dataset_rows = self._load_dataset_rows()

    def _load_dataset_rows(self) -> list[dict[str, str]]:
        dataset_path = PROJECT_DIR / "Cleaned_Indian_Food_Dataset.csv"
        if not dataset_path.exists():
            return []

        rows: list[dict[str, str]] = []
        try:
            with dataset_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    rows.append({str(key): str(value or "") for key, value in row.items()})
        except Exception:
            return []
        return rows

    def _split_csv_ingredients(self, text: str) -> list[str]:
        if not text.strip():
            return []
        items = [part.strip() for part in text.split(",") if part.strip()]
        cleaned: list[str] = []
        for item in items:
            normalized = item.lower().strip()
            normalized = re.sub(r"\([^)]*\)", "", normalized)
            normalized = normalized.replace("  ", " ").strip(" -")
            if normalized:
                cleaned.append(normalized)
        return normalize_text_list(cleaned)

    def _clean_ingredient_name(self, text: str) -> str:
        name = text.strip().lower()
        name = re.sub(r"\([^)]*\)", "", name)
        if " - " in name:
            name = name.split(" - ", 1)[0].strip()
        name = re.sub(r"\b(?:finely|thinly|roughly|coarsely|chopped|sliced|diced|minced|grated|crushed|fresh|optional)\b", "", name)
        name = re.sub(r"\s+", " ", name).strip(" ,.-")
        return name

    def _parse_dataset_ingredient_entry(self, raw_item: str) -> dict[str, str] | None:
        raw = raw_item.strip()
        if not raw:
            return None

        lowered = re.sub(r"\([^)]*\)", "", raw.lower()).strip()
        lowered = re.sub(r"\s+", " ", lowered)

        quantity_pattern = (
            r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?|a|an|few|pinch)"
            r"(?:\s*(?:to|-)\s*(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?))?"
        )
        unit_pattern = (
            r"(?:tsp|tbsp|teaspoon|teaspoons|tablespoon|tablespoons|cup|cups|gram|grams|g|kg|kgs|ml|l|"
            r"litre|litres|nos|no|piece|pieces|clove|cloves|sprig|sprigs|inch|inches)"
        )

        measurement = ""
        ingredient_text = lowered

        qty_match = re.match(rf"^\s*({quantity_pattern}(?:\s*{unit_pattern})?)\s+(.+)$", lowered)
        if qty_match:
            measurement = qty_match.group(1).strip()
            ingredient_text = qty_match.group(2).strip()
        elif " - " in lowered:
            left, right = [segment.strip() for segment in lowered.split(" - ", 1)]
            ingredient_text = left
            measurement = right if right in {"to taste", "as required"} else ""

        ingredient = self._clean_ingredient_name(ingredient_text)
        if not ingredient:
            return None

        normalized = f"{measurement} {ingredient}".strip() if measurement else ingredient
        return {
            "raw": lowered,
            "ingredient": ingredient,
            "measurement": measurement,
            "normalized": normalized,
        }

    def _parse_dataset_ingredient_entries(
        self,
        translated_ingredients: str,
        fallback_ingredients: list[str],
    ) -> list[dict[str, str]]:
        parts = [part.strip() for part in translated_ingredients.split(",") if part.strip()]
        entries: list[dict[str, str]] = []

        for part in parts:
            parsed = self._parse_dataset_ingredient_entry(part)
            if parsed is not None:
                entries.append(parsed)

        if not entries:
            for ingredient in fallback_ingredients:
                ing = self._clean_ingredient_name(ingredient)
                if ing:
                    entries.append(
                        {
                            "raw": ing,
                            "ingredient": ing,
                            "measurement": self._measurement_for(ing),
                            "normalized": f"{self._measurement_for(ing)} {ing}".strip(),
                        }
                    )

        dedup: dict[str, dict[str, str]] = {}
        for entry in entries:
            ingredient = entry["ingredient"].strip().lower()
            if ingredient and ingredient not in dedup:
                dedup[ingredient] = entry
        return list(dedup.values())

    def _ingredient_matches_user_input(self, recipe_ingredient: str, user_ingredients: set[str]) -> bool:
        target = recipe_ingredient.strip().lower()
        if not target:
            return False
        if target in user_ingredients:
            return True

        for user_item in user_ingredients:
            if target in user_item or user_item in target:
                if len(target) > 2 and len(user_item) > 2:
                    return True
        return False

    def _build_extra_ingredient_strings(
        self,
        missing_ingredients: list[str],
        rows: list[dict[str, str]],
    ) -> list[str]:
        measurement_map: dict[str, str] = {}
        for row in rows:
            ingredient = str(row.get("ingredient", "")).strip().lower()
            measurement = str(row.get("measurement", "")).strip()
            if ingredient and ingredient not in measurement_map:
                measurement_map[ingredient] = measurement

        extras: list[str] = []
        for ingredient in missing_ingredients:
            measurement = measurement_map.get(ingredient, "")
            extra_text = f"{measurement} {ingredient}".strip() if measurement else ingredient
            if extra_text:
                extras.append(extra_text)
        return extras

    def _parse_dish_query(self, dish_name: str) -> tuple[str, list[str]]:
        query = dish_name.strip().lower()
        query = re.sub(r"[^a-z0-9\s-]", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        stopwords = {"recipe", "style", "and", "with", "from", "of", "the", "a", "an", "for"}
        tokens = [token for token in query.split() if token not in stopwords and len(token) > 1]
        if not tokens:
            return "", []

        # Canonical dish token is the last meaningful token (e.g., "spinach dosa" -> "dosa").
        dish_token = tokens[-1]
        variant_tokens = [token for token in tokens[:-1] if token != dish_token]
        return dish_token, variant_tokens

    def _row_title(self, row: dict[str, str]) -> str:
        for key in ("TranslatedRecipeName", "recipe_name", "name", "RecipeName", "title"):
            value = str(row.get(key, "")).strip().lower()
            if value:
                return value
        return ""

    def _extract_user_dish_query(self, preferences: RecipePreferences) -> str:
        explicit = str(preferences.dish_name or "").strip().lower()
        if explicit:
            return self._clean_user_query_text(explicit)

        raw = str(preferences.user_text or "").strip().lower()
        if not raw:
            return ""
        cleaned = self._clean_user_query_text(raw)
        if cleaned:
            return cleaned

        raw = re.sub(r"[^a-z0-9\s-]", " ", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        tokens = [token for token in raw.split() if token not in GENERIC_QUERY_TOKENS and token not in FILLER_WORDS]
        if len(tokens) >= 2:
            return " ".join(tokens[-3:]).strip()
        return raw

    def _clean_user_query_text(self, raw_text: str) -> str:
        text = re.sub(r"[^a-z0-9\s-]", " ", str(raw_text or "").lower())
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        tokens = [token for token in text.split() if token not in FILLER_WORDS]
        if not tokens:
            return ""

        stop_at = {"with", "using", "from", "for", "please", "today", "now"}
        cleaned_tokens: list[str] = []
        for token in tokens:
            if token in stop_at:
                break
            cleaned_tokens.append(token)

        return " ".join(cleaned_tokens).strip()

    def _extract_intent_parts(self, cleaned_query: str) -> tuple[str, str]:
        tokens = [token for token in cleaned_query.split() if token]
        if not tokens:
            return "", ""

        dish_type = ""
        for token in reversed(tokens):
            if token in DISH_TYPES:
                dish_type = token
                break

        main_ingredient = ""
        for token in tokens:
            if token == dish_type:
                continue
            if token not in GENERIC_QUERY_TOKENS and token not in FILLER_WORDS:
                main_ingredient = token
                break

        return main_ingredient, dish_type

    def _is_unrelated_variant_title(self, title: str, query: str) -> bool:
        for keyword in UNRELATED_VARIANT_KEYWORDS:
            if keyword in title and keyword not in query:
                return True
        return False

    def _clean_recipe_title(self, title: str) -> str:
        """PHASE 2: Clean recipe title by removing extra descriptors and punctuation.
        
        Example:
          'Vegetable Sizzler Recipe With Potato Tikki, Mint Rice & Corn' 
          → 'vegetable sizzler potato tikki mint rice corn'
        """
        cleaned = title.lower()
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)  # Remove punctuation
        cleaned = re.sub(r"\s+", " ", cleaned).strip()  # Normalize spaces
        
        # Remove descriptor keywords
        words = cleaned.split()
        words = [w for w in words if w not in DESCRIPTOR_KEYWORDS and len(w) > 2]
        
        return " ".join(words)

    def _is_composite_recipe(self, title: str, query: str) -> bool:
        """PHASE 4: Check if recipe is a composite/fancy dish.
        
        Characteristics:
        - Contains words: sizzler, platter, combo, with, &, sauce
        - Unless explicitly in user query
        """
        for keyword in COMPOSITE_RECIPE_KEYWORDS:
            if keyword in title and keyword not in query:
                if keyword not in {"with", "and"}:  # Allow 'with' and 'and' in some contexts
                    return True
        return False

    def _query_aliases(self, query: str) -> list[str]:
        if not query:
            return []

        aliases = {query}
        for canonical, variants in DISH_ALIAS_MAP.items():
            all_terms = {canonical, *variants}
            if query in all_terms:
                aliases.update(all_terms)
        return [alias for alias in aliases if alias.strip()]

    def _query_features(self, query: str) -> tuple[str, str]:
        if not query:
            return "", ""

        tokens = [token for token in query.split() if token]
        dish_type = next((token for token in tokens if token in DISH_TYPES), "")
        main_keyword = next(
            (
                token for token in tokens
                if token in MAJOR_INGREDIENTS or (token not in DISH_TYPES and token not in GENERIC_QUERY_TOKENS and len(token) > 2)
            ),
            "",
        )
        return dish_type, main_keyword

    def _score_title_match(
        self,
        title: str,
        query: str,
        aliases: list[str],
        dish_type: str,
        main_keyword: str,
        input_ingredients: list[str],
        row: dict[str, str],
    ) -> int:
        if not title:
            return -999

        score = 0

        exact_phrase_hit = bool(query and len(query) > 3 and query in title)
        alias_phrase_hit = any(alias for alias in aliases if alias != query and len(alias) > 3 and alias in title)
        if exact_phrase_hit:
            score += 5
        elif alias_phrase_hit:
            score += 3

        if main_keyword and main_keyword in title:
            score += 2

        if dish_type and dish_type in title:
            score += 2

        if self._is_unrelated_variant_title(title, query):
            score -= 5

        if dish_type:
            other_type = next((token for token in DISH_TYPES if token != dish_type and token in title), "")
            if other_type:
                score -= 3

        if main_keyword:
            conflicting_major = next((token for token in MAJOR_INGREDIENTS if token != main_keyword and token in title), "")
            if conflicting_major and main_keyword not in title:
                score -= 2

        recipe_ingredients = self._split_csv_ingredients(row.get("Cleaned-Ingredients", ""))
        if not recipe_ingredients:
            recipe_ingredients = self._split_csv_ingredients(row.get("TranslatedIngredients", ""))
        if set(normalize_text_list(input_ingredients)) & set(recipe_ingredients):
            score += 1

        return score

    def _rank_recipe_candidates(
        self,
        candidates: list[dict[str, str]],
        input_ingredients: list[str],
        preferred_tokens: list[str],
    ) -> dict[str, str] | None:
        if not candidates:
            return None

        user_set = set(normalize_text_list(input_ingredients))
        token_set = set(preferred_tokens)
        scored: list[tuple[int, int, int, dict[str, str]]] = []

        for row in candidates:
            recipe_ingredients = self._split_csv_ingredients(row.get("Cleaned-Ingredients", ""))
            if not recipe_ingredients:
                recipe_ingredients = self._split_csv_ingredients(row.get("TranslatedIngredients", ""))
            recipe_set = set(recipe_ingredients)

            match_count = len(user_set & recipe_set)
            irrelevant_count = len(recipe_set - user_set)
            key_token_hits = 0
            if token_set:
                key_token_hits = len({tok for tok in token_set if tok in recipe_set})

            # Final filter: if user gave ingredients, reject clearly unrelated candidates.
            if user_set:
                if match_count == 0:
                    continue
                if irrelevant_count > (match_count + 8):
                    continue

            scored.append((match_count, key_token_hits, -irrelevant_count, row))

        if not scored:
            return None

        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return scored[0][3]

    def _find_dataset_row(self, dish_name: str, input_ingredients: list[str]) -> tuple[dict[str, str] | None, str]:
        """PHASE 3-7: Find best recipe from dataset using 9-phase pipeline.
        
        Phases:
        1. Parse user query intent
        2. Clean recipe titles
        3. Filter by dish type
        4. Detect & penalize composite recipes
        5. Score by relevance
        6. Compare ingredients
        7. Apply threshold
        """
        if not dish_name.strip() or not self.dataset_rows:
            return None, "none"

        # PHASE 1: Parse user query intent
        raw_query = re.sub(r"\s+", " ", dish_name.strip().lower())
        cleaned_query = self._clean_user_query_text(raw_query)
        main_ingredient, dish_type = self._extract_intent_parts(cleaned_query)

        print(f"[phase1_intent] raw_query={raw_query}")
        print(f"[phase1_intent] cleaned_query={cleaned_query}")
        print(f"[phase1_intent] main_ingredient={main_ingredient}")
        print(f"[phase1_intent] dish_type={dish_type}")

        if not cleaned_query or not dish_type:
            return None, "dataset_no_match"

        query = cleaned_query
        aliases = self._query_aliases(query)
        allowed_dish_types = {token for alias in aliases for token in alias.split() if token in DISH_TYPES}
        allowed_dish_types.update(DISH_TYPE_EQUIVALENTS.get(dish_type, set()))
        if dish_type:
            allowed_dish_types.add(dish_type)

        if not aliases:
            return None, "dataset_no_match"

        scored_rows: list[tuple[int, bool, dict[str, str]]] = []
        for row in self.dataset_rows:
            title = self._row_title(row)
            if not title:
                continue

            # PHASE 2: Clean recipe title for accurate matching
            cleaned_title = self._clean_recipe_title(title)

            # PHASE 3: Hard dish type filter
            if allowed_dish_types and not any(token in cleaned_title for token in allowed_dish_types):
                continue

            if self._is_unrelated_variant_title(cleaned_title, query):
                continue

            # PHASE 4: Check for composite/fancy recipes
            is_composite = self._is_composite_recipe(cleaned_title, query)

            # PHASE 5: Calculate title score
            exact_phrase_hit = bool(query and len(query) > 3 and query in cleaned_title)
            alias_phrase_hit = any(alias for alias in aliases if alias != query and len(alias) > 3 and alias in cleaned_title)
            main_in_title = bool(main_ingredient and main_ingredient in cleaned_title)

            # PHASE 6: Ingredient matching
            recipe_ingredients = self._split_csv_ingredients(row.get("Cleaned-Ingredients", ""))
            if not recipe_ingredients:
                recipe_ingredients = self._split_csv_ingredients(row.get("TranslatedIngredients", ""))
            main_in_ingredients = bool(main_ingredient and any(main_ingredient in ing for ing in recipe_ingredients))
            user_ingredient_set = set(normalize_text_list(input_ingredients))
            ingredient_overlap = bool(user_ingredient_set & set(recipe_ingredients))

            # Main ingredient must influence selection for bath/rice family and similar dishes.
            if main_ingredient and not (main_in_title or main_in_ingredients):
                continue

            # PHASE 5: Calculate score
            score = 0
            
            # Exact match: +10
            if exact_phrase_hit:
                score += 10
            # Alias match: +6 (PHASE 5 spec: "+6 contains main ingredient in title")
            elif alias_phrase_hit:
                score += 6
            
            # Main ingredient in title: +6
            if main_in_title:
                score += 6
            
            # Dish type match: +5
            if dish_type and any(token in cleaned_title for token in allowed_dish_types):
                score += 5
            
            # Ingredient overlap: +3
            if ingredient_overlap:
                score += 3
            
            # PHASE 4: Composite recipe penalty
            if is_composite:
                score -= 10
            
            # Unrelated major ingredient penalty: -5
            if self._is_unrelated_variant_title(cleaned_title, query):
                score -= 10
            else:
                # Check for unrelated major ingredients
                has_unrelated_major = False
                for major_ing in MAJOR_INGREDIENTS:
                    if major_ing in cleaned_title and major_ing != main_ingredient:
                        if not any(major_ing in ing for ing in recipe_ingredients):
                            has_unrelated_major = True
                            break
                if has_unrelated_major:
                    score -= 5

            scored_rows.append((score, exact_phrase_hit, row))

        if not scored_rows:
            print("[phase7_threshold] dataset_candidates=0, threshold_fail")
            return None, "dataset_no_match"

        scored_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score, exact_phrase_hit, best_row = scored_rows[0]
        print(f"[phase7_threshold] dataset_candidates={len(scored_rows)}, best_score={best_score}")

        # PHASE 7: Dataset threshold (must be >= 7 for dataset acceptance)
        if best_score < 7:
            print(f"[phase7_threshold] score_below_threshold (need >=7, got {best_score})")
            return None, "dataset_no_match"

        source = "dataset_exact" if exact_phrase_hit else "dataset_similarity"
        return best_row, source

    def _parse_measured_ingredient_rows(self, translated_ingredients: str, fallback_ingredients: list[str]) -> list[dict[str, str]]:
        parts = [part.strip() for part in translated_ingredients.split(",") if part.strip()]
        rows: list[dict[str, str]] = []
        unit_pattern = r"(?:tsp|tbsp|teaspoon|teaspoons|tablespoon|tablespoons|cup|cups|gram|grams|kg|ml|l|inch|inches|sprig|sprigs|clove|cloves|pinch|nos|no)"

        for part in parts:
            raw = part.strip()
            lower = raw.lower()

            if " - " in lower and not re.match(r"^\s*\d", lower):
                left, right = [seg.strip() for seg in lower.split(" - ", 1)]
                ingredient = re.sub(r"\([^)]*\)", "", left).strip()
                measurement = "to taste" if "taste" in right else self._measurement_for(ingredient)
                if ingredient:
                    rows.append({"ingredient": ingredient, "measurement": self._enforce_measurement_caps(ingredient, measurement)})
                continue

            match = re.match(rf"^\s*([\d\s/.-]+\s*{unit_pattern}?)\s+(.*)$", lower)
            if match:
                measurement = match.group(1).strip()
                ingredient = re.sub(r"\([^)]*\)", "", match.group(2)).strip(" -")
                if ingredient:
                    rows.append({"ingredient": ingredient, "measurement": self._enforce_measurement_caps(ingredient, measurement)})
            else:
                ingredient = re.sub(r"\([^)]*\)", "", lower).strip(" -")
                if ingredient:
                    rows.append({"ingredient": ingredient, "measurement": self._enforce_measurement_caps(ingredient, self._measurement_for(ingredient))})

        if not rows:
            rows = [{"ingredient": ing, "measurement": self._enforce_measurement_caps(ing, self._measurement_for(ing))} for ing in fallback_ingredients]

        dedup: dict[str, str] = {}
        for row in rows:
            ing = row["ingredient"].strip().lower()
            if ing and ing not in dedup:
                dedup[ing] = row["measurement"]
        return [{"ingredient": ing, "measurement": meas} for ing, meas in dedup.items()]

    def _steps_from_instructions(self, instructions: str) -> list[str]:
        blob = instructions.replace("\r", "\n").strip()
        if not blob:
            return []
        chunks = [seg.strip() for seg in re.split(r"\n+|(?<=[.!?])\s+", blob) if seg.strip()]
        steps: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            normalized = chunk if chunk.endswith((".", "!", "?")) else f"{chunk}."
            steps.append(f"{idx}. {normalized}")
        return steps

    def _apply_health_goals_to_rows(
        self,
        rows: list[dict[str, str]],
        full_ingredients: list[str],
        preferences: RecipePreferences,
    ) -> tuple[list[dict[str, str]], list[str], str]:
        goals = {goal.strip().lower() for goal in preferences.health_goals}
        health_note = ", ".join(sorted(goals)) if goals else "none"

        adjusted_rows = rows
        adjusted_full = full_ingredients

        if "diabetic friendly" in goals:
            banned = {"sugar", "jaggery", "honey"}
            adjusted_rows = [r for r in adjusted_rows if r["ingredient"] not in banned]
            adjusted_full = [ing for ing in adjusted_full if ing not in banned]

        if "weight loss" in goals:
            for row in adjusted_rows:
                if row["ingredient"] in {"oil", "ghee", "butter", "sunflower oil", "olive oil", "mustard oil"}:
                    row["measurement"] = self._enforce_measurement_caps(row["ingredient"], "1 tsp")

        return adjusted_rows, adjusted_full, health_note

    def _generate_from_dataset(self, ingredients: list[str], preferences: RecipePreferences) -> dict[str, Any] | None:
        row: dict[str, str] | None = None
        source = "none"

        dish_query = self._extract_user_dish_query(preferences)
        if dish_query:
            row, source = self._find_dataset_row(dish_query, ingredients)
        else:
            # Ingredient-only mode: still prefer dataset over generic fallback/model.
            ranked = self._rank_recipe_candidates(self.dataset_rows, ingredients, [])
            if ranked is not None:
                row = ranked
                source = "dataset_similarity"

        if row is None:
            return None

        input_ingredients = normalize_text_list(ingredients)
        full_ingredients = self._split_csv_ingredients(row.get("Cleaned-Ingredients", ""))
        if not full_ingredients:
            full_ingredients = self._split_csv_ingredients(row.get("TranslatedIngredients", ""))

        entries = self._parse_dataset_ingredient_entries(row.get("TranslatedIngredients", ""), full_ingredients)
        rows = [
            {
                "ingredient": entry["ingredient"],
                "measurement": entry["measurement"] or self._measurement_for(entry["ingredient"]),
            }
            for entry in entries
        ]

        rows, adjusted_full_ingredients, health_note = self._apply_health_goals_to_rows(rows, full_ingredients, preferences)
        adjusted_rows = [
            {
                "ingredient": str(r.get("ingredient", "")).strip().lower(),
                "measurement": str(r.get("measurement", "")).strip(),
            }
            for r in rows
            if str(r.get("ingredient", "")).strip()
        ]

        user_set = set(input_ingredients)
        matched_ingredients = [
            row_item["ingredient"]
            for row_item in adjusted_rows
            if self._ingredient_matches_user_input(row_item["ingredient"], user_set)
        ]
        missing_ingredients = [
            row_item["ingredient"]
            for row_item in adjusted_rows
            if row_item["ingredient"] not in set(matched_ingredients)
        ]
        extra_ingredients = self._build_extra_ingredient_strings(missing_ingredients, adjusted_rows)

        steps = self._steps_from_instructions(row.get("TranslatedInstructions", ""))
        if not steps:
            steps = self._normalize_steps([], preferences)

        total_minutes = row.get("TotalTimeInMins", "").strip()
        cooking_time = f"{total_minutes} minutes" if total_minutes.isdigit() else self._estimate_cooking_time(preferences, rows)

        return {
            "recipe_name": row.get("TranslatedRecipeName", "").strip() or preferences.dish_name.strip(),
            "input_ingredients": input_ingredients,
            "matched_ingredients": matched_ingredients,
            "missing_ingredients": missing_ingredients,
            "extra_ingredients": extra_ingredients,
            "full_ingredients": [row_item["ingredient"] for row_item in adjusted_rows] or adjusted_full_ingredients,
            "ingredients": [row_item["ingredient"] for row_item in adjusted_rows] or adjusted_full_ingredients,
            "ingredients_with_measurements": adjusted_rows,
            "steps": steps,
            "cooking_time": cooking_time,
            "servings": str(preferences.servings),
            "health_optimization": health_note,
            "meal_type": preferences.meal_type,
            "nutrition": self._normalize_nutrition({}, adjusted_rows, preferences.servings),
            "source": source,
            "engine_used": "csv",
            "generation_source": source,
        }

    def build_prompt(self, ingredients: list[str], preferences: RecipePreferences) -> str:
        clean_ingredients = normalize_text_list(ingredients)
        requested_dish = self._extract_user_dish_query(preferences) or "none"
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
        prompt = (PROMPT_TEMPLATE.format(
            ingredients=", ".join(clean_ingredients) if clean_ingredients else "none",
            requested_dish=requested_dish,
            preferences=pref_blob,
            health_goals=", ".join(preferences.health_goals) if preferences.health_goals else "none",
        ) + "\n" + CHEF_RULES_APPENDIX)

        prompt += "\n" + STRICT_REALISM_RULES
        
        # Add health goals rules if any are specified
        if preferences.health_goals:
            prompt += "\n" + HEALTH_GOALS_RULES
        
        return prompt

    def generate(self, ingredients: list[str], preferences: RecipePreferences) -> dict[str, Any]:
        clean_ingredients = normalize_text_list(ingredients)

        dataset_recipe = self._generate_from_dataset(clean_ingredients, preferences)
        if dataset_recipe is not None:
            return dataset_recipe

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
        full_ingredients = normalize_text_list([row["ingredient"] for row in rows])
        matched_ingredients = [item for item in input_ingredients if item in set(full_ingredients)]
        missing_ingredients = [item for item in full_ingredients if item not in set(matched_ingredients)]
        measured_extras = self._build_extra_ingredient_strings(missing_ingredients, rows)

        model_steps = recipe.get("steps", [])
        if not model_steps:
            model_steps = recipe.get("instructions", [])

        numbered_steps = self._normalize_steps(model_steps, preferences)
        nutrition = self._normalize_nutrition(self._extract_nutrition_payload(recipe), rows, preferences.servings)

        return {
            "recipe_name": str(recipe.get("recipe_name") or self._fallback_name(input_ingredients, preferences)),
            "input_ingredients": input_ingredients,
            "matched_ingredients": matched_ingredients,
            "missing_ingredients": missing_ingredients,
            "full_ingredients": full_ingredients,
            "ingredients": full_ingredients,
            "extra_ingredients": measured_extras,
            "health_optimization": ", ".join(preferences.health_goals) if preferences.health_goals else "none",
            "ingredients_with_measurements": rows,
            "steps": numbered_steps,
            "cooking_time": str(recipe.get("cooking_time") or self._estimate_cooking_time(preferences, rows)),
            "servings": str(recipe.get("servings") or preferences.servings),
            "meal_type": preferences.meal_type,
            "nutrition": nutrition,
            "source": "model",
            "engine_used": "qwen",
            "generation_source": "model",
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

        if ("dal" in ing or "toor dal" in ing or "lentils" in ing) and "tamarind" in ing:
            return "South Indian Sambar"
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

        if ingredient == "salt":
            return "to taste"
        if ingredient in {"cumin seeds", "mustard seeds"}:
            return "1/2 tsp"
        if ingredient in {"turmeric", "turmeric powder"}:
            return "1/4 tsp"
        if ingredient in {"chili powder", "chilli powder", "red chilli powder", "red chili powder"}:
            return "1/2 tsp"
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
        if ingredient in {"onion", "tomato", "potato", "carrot", "capsicum", "brinjal", "beans", "drumstick"}:
            return "1 medium"
        if ingredient in {"rice", "flour", "maida", "atta", "semolina", "oats", "quinoa"}:
            return "1 cup"
        if ingredient in {"chicken", "fish", "mutton", "paneer"}:
            return "250 grams"
        if ingredient in {"egg", "boiled egg", "fried egg"}:
            return "2 nos"
        return "1 cup"

    def _to_float(self, token: str) -> float | None:
        value = token.strip().lower()
        if not value:
            return None

        fraction_map = {
            "1/4": 0.25,
            "1/3": 0.333,
            "1/2": 0.5,
            "2/3": 0.667,
            "3/4": 0.75,
        }
        if value in fraction_map:
            return fraction_map[value]
        try:
            return float(value)
        except Exception:
            return None

    def _enforce_measurement_caps(self, ingredient: str, measurement: str) -> str:
        ing = ingredient.strip().lower()
        text = measurement.strip().lower()

        if ing == "salt":
            return "to taste"

        if ing == "cumin seeds":
            return "1 tsp" if "1 tsp" in text else "1/2 tsp"
        if ing == "mustard seeds":
            return "1 tsp" if "1 tsp" in text else "1/2 tsp"
        if ing in {"turmeric", "turmeric powder"}:
            return "1/2 tsp" if "1/2" in text else "1/4 tsp"
        if ing in {"chili powder", "chilli powder", "red chilli powder", "red chili powder"}:
            return "1 tsp" if "1 tsp" in text else "1/2 tsp"

        if ing in {"water", "milk", "oil", "ghee", "butter"}:
            match = re.search(r"(\d+(?:\.\d+)?|\d+/\d+)\s*(cup|cups|tbsp|tsp)", text)
            qty = self._to_float(match.group(1)) if match else None
            unit = match.group(2) if match else ""

            if ing == "water":
                if qty is None or unit not in {"cup", "cups"}:
                    return "2 cups"
                return f"{max(1, min(3, int(round(qty))))} cups"

            if ing == "milk":
                if qty is None or unit not in {"cup", "cups"}:
                    return "1 cup"
                return f"{max(1, min(2, int(round(qty))))} cups"

            if qty is None:
                return "1 tbsp"
            if unit == "tsp":
                tbsp_equiv = qty / 3
            else:
                tbsp_equiv = qty
            tbsp_equiv = max(0.5, min(2.0, tbsp_equiv))
            if abs(tbsp_equiv - 0.5) < 0.01:
                return "1/2 tbsp"
            if abs(tbsp_equiv - 1.0) < 0.01:
                return "1 tbsp"
            if abs(tbsp_equiv - 1.5) < 0.01:
                return "1 1/2 tbsp"
            return "2 tbsp"

        if ing in {"onion", "tomato", "potato", "carrot", "capsicum", "brinjal", "beans", "drumstick"}:
            match = re.search(r"(\d+(?:\.\d+)?)", text)
            qty = float(match.group(1)) if match else 1.0
            qty = max(1, min(3, int(round(qty))))
            return f"{qty} medium"

        return measurement or self._measurement_for(ing)

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
                    resolved_measurement = measurement or self._measurement_for(ingredient)
                    normalized.append(
                        {
                            "ingredient": ingredient,
                            "measurement": self._enforce_measurement_caps(ingredient, resolved_measurement),
                        }
                    )

        required = normalize_text_list(input_ingredients + extra_ingredients)
        existing = {r["ingredient"] for r in normalized}
        for ingredient in required:
            if ingredient not in existing:
                base = self._measurement_for(ingredient)
                normalized.append({"ingredient": ingredient, "measurement": self._enforce_measurement_caps(ingredient, base)})

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
        active_prep = 10
        passive_time = 0
        cooking_time = 20

        if preferences.meal_type == "dessert":
            cooking_time = 25
        if preferences.meal_type in {"breakfast", "snack"}:
            cooking_time = 15
        if len(rows) > 8:
            active_prep += 5
            cooking_time += 10

        # Common passive stages for Indian cooking styles.
        if any(goal.lower() in {"high protein", "muscle gain"} for goal in preferences.health_goals):
            passive_time = max(passive_time, 15)

        total_time = active_prep + passive_time + cooking_time
        return (
            f"Active Prep: {active_prep} minutes | "
            f"Passive: {passive_time} minutes | "
            f"Cooking: {cooking_time} minutes | "
            f"Total: {total_time} minutes"
        )

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
        full_ingredients = normalize_text_list([row["ingredient"] for row in rows])
        matched_ingredients = [item for item in input_ingredients if item in set(full_ingredients)]
        missing_ingredients = [item for item in full_ingredients if item not in set(matched_ingredients)]
        measured_extras = self._build_extra_ingredient_strings(missing_ingredients, rows)

        return {
            "recipe_name": self._fallback_name(input_ingredients, preferences),
            "input_ingredients": input_ingredients,
            "matched_ingredients": matched_ingredients,
            "missing_ingredients": missing_ingredients,
            "full_ingredients": full_ingredients,
            "ingredients": full_ingredients,
            "extra_ingredients": measured_extras,
            "health_optimization": ", ".join(preferences.health_goals) if preferences.health_goals else "none",
            "ingredients_with_measurements": rows,
            "steps": steps,
            "cooking_time": self._estimate_cooking_time(preferences, rows),
            "servings": str(preferences.servings),
            "meal_type": preferences.meal_type,
            "nutrition": self._normalize_nutrition({}, rows, preferences.servings),
            "source": "fallback",
            "engine_used": "template",
            "generation_source": "fallback",
        }


def parse_text_hint_to_ingredients(text: str, known_ingredients: list[str]) -> list[str]:
    clean_text = text.lower().strip()
    if not clean_text:
        return []
    detected = [item for item in known_ingredients if item in clean_text]
    return normalize_text_list(detected)
