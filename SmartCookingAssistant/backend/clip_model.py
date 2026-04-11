from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .config import settings
from .utils import normalize_text_list

ingredients_data = {
    "vegetables": [
        "tomato", "potato", "onion", "garlic", "ginger", "carrot", "cabbage",
        "cauliflower", "capsicum", "green chilli", "okra", "spinach",
        "coriander leaves", "mint leaves", "beetroot", "pumpkin", "sweet potato",
        "peas", "beans", "mushroom", "broccoli", "lettuce", "zucchini", "radish",
        "turnip", "leeks", "spring onion", "drumstick", "brinjal", "eggplant"
    ],
    "fruits": [
        "apple", "banana", "mango", "orange", "papaya", "pineapple", "watermelon",
        "grapes", "pomegranate", "kiwi", "strawberry", "blueberry", "pear", "peach",
        "plum", "cherry", "dragon fruit", "avocado", "coconut", "guava", "sapota"
    ],
    "spices": [
        "turmeric", "red chilli powder", "black pepper", "cumin seeds", "mustard seeds",
        "coriander powder", "garam masala", "cardamom", "cloves", "cinnamon",
        "bay leaf", "asafoetida", "fenugreek seeds", "fennel seeds",
        "star anise", "nutmeg", "mace"
    ],
    "grains": [
        "rice", "wheat", "flour", "atta", "maida", "semolina", "oats", "corn", "millets",
        "barley", "quinoa"
    ],
    "pulses": [
        "dal", "lentils", "chickpeas", "green gram", "black gram",
        "kidney beans", "toor dal", "urad dal", "moong dal"
    ],
    "nonveg": [
        "chicken", "fish", "mutton", "egg", "boiled egg", "fried egg",
        "prawns", "crab"
    ],
    "dairy": [
        "milk", "curd", "yogurt", "butter", "ghee", "paneer", "cheese",
        "cream", "buttermilk", "condensed milk"
    ],
    "cooking_basics": [
        "salt", "sugar", "jaggery", "honey",
        "oil", "sunflower oil", "olive oil", "mustard oil",
        "vinegar", "soy sauce", "tomato ketchup"
    ]
}

PROMPT_TEMPLATES = [
    "a photo of {ingredient}",
    "fresh {ingredient}",
    "close-up of {ingredient}",
]

# Extra phrase variants improve CLIP grounding for pantry staples that are often confused.
INGREDIENT_PROMPT_VARIANTS: dict[str, list[str]] = {
    "tomato": [
        "ripe red tomato",
        "fresh tomato",
        "whole tomato",
    ],
    "onion": [
        "red onion",
        "whole onion bulb",
        "fresh onion",
    ],
    "dal": [
        "split yellow lentils",
        "raw dal in a bowl",
        "uncooked lentils",
    ],
    "toor dal": ["pigeon peas lentils", "arhar dal", "split toor lentils"],
    "moong dal": ["split mung lentils", "yellow moong dal", "mung dal in bowl"],
    "urad dal": ["split black gram", "white urad dal", "urad lentils"],
    "lentils": ["mixed lentils", "dry lentils in bowl"],
}

DAL_FAMILY = {"dal", "lentils", "toor dal", "moong dal", "urad dal", "green gram", "black gram"}


@lru_cache(maxsize=1)
def get_clip_components() -> tuple[CLIPModel, CLIPProcessor, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(settings.clip_model_name)
    processor = CLIPProcessor.from_pretrained(settings.clip_model_name)
    model.to(device)
    if device.type == "cuda":
        model = model.half()
    model.eval()
    return model, processor, device


class IngredientDetector:
    def __init__(self) -> None:
        self.model, self.processor, self.device = get_clip_components()
        self.ingredients = self._build_ingredient_list()
        self._prompts, self._prompt_labels = self._build_prompts()
        self._text_features = self._build_text_features(self._prompts)

    def _build_ingredient_list(self) -> list[str]:
        ordered: list[str] = []
        for group in ingredients_data.values():
            ordered.extend(group)
        return normalize_text_list(ordered)

    def _build_prompts(self) -> tuple[list[str], list[str]]:
        prompts: list[str] = []
        labels: list[str] = []
        for ingredient in self.ingredients:
            for template in PROMPT_TEMPLATES:
                prompts.append(template.format(ingredient=ingredient))
                labels.append(ingredient)
            for variant in INGREDIENT_PROMPT_VARIANTS.get(ingredient, []):
                prompts.append(f"a photo of {variant}")
                labels.append(ingredient)
        return prompts, labels

    def _build_text_features(self, prompts: list[str]) -> torch.Tensor:
        text_inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}

        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            # Use the CLIP projection head explicitly so text/image features always share dims.
            text_outputs = self.model.text_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
                return_dict=True,
            )
            text_features = self.model.text_projection(text_outputs.pooler_output)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        return text_features

    def _estimate_red_food_likelihood(self, image: Image.Image) -> float:
        rgb = image.convert("RGB")
        pixels = list(rgb.getdata())
        if not pixels:
            return 0.0

        red_like = 0
        for r, g, b in pixels:
            if r > 80 and r > (g * 1.2) and r > (b * 1.2):
                red_like += 1
        return red_like / len(pixels)

    def _correct_common_misclassifications(
        self,
        ranked: list[dict[str, Any]],
        aggregate_scores: defaultdict[str, list[float]],
        red_likelihood: float,
    ) -> list[dict[str, Any]]:
        by_name = {item["ingredient"]: item for item in ranked}
        radish_item = by_name.get("radish")
        tomato_scores = aggregate_scores.get("tomato", [])
        tomato_score = (sum(tomato_scores) / len(tomato_scores)) if tomato_scores else 0.0

        if radish_item:
            radish_score = float(radish_item.get("confidence", 0.0))
            # Tomato images are often red-dominant. If tomato score is close and image is red,
            # prefer tomato over radish.
            tomato_close_match = tomato_score > 0 and red_likelihood >= 0.08 and tomato_score + 0.015 >= radish_score
            low_conf_radish_case = radish_score <= 0.20 and tomato_score >= 0.08 and red_likelihood >= 0.04
            if tomato_close_match or low_conf_radish_case:
                by_name.pop("radish", None)
                existing_tomato = by_name.get("tomato")
                boosted = max(tomato_score, radish_score)
                if existing_tomato is None:
                    by_name["tomato"] = {
                        "ingredient": "tomato",
                        "confidence": boosted,
                    }
                else:
                    existing_tomato["confidence"] = max(float(existing_tomato.get("confidence", 0.0)), boosted)

        radish_item = by_name.get("radish")
        if not radish_item:
            corrected = list(by_name.values())
            corrected.sort(key=lambda item: item["confidence"], reverse=True)
            return corrected

        # If dal-like labels are present in near-top candidates, prefer them over radish
        # for pantry dry-lentil images where CLIP commonly confuses texture/shape.
        best_dal_label = ""
        best_dal_score = 0.0
        for label in DAL_FAMILY:
            scores = aggregate_scores.get(label, [])
            if not scores:
                continue
            score = sum(scores) / len(scores)
            if score > best_dal_score:
                best_dal_score = score
                best_dal_label = label

        radish_score = float(radish_item.get("confidence", 0.0))
        if best_dal_label and best_dal_score >= (settings.min_confidence * 0.6) and best_dal_score + 0.08 >= radish_score:
            by_name.pop("radish", None)
            existing = by_name.get(best_dal_label)
            if existing is None:
                by_name[best_dal_label] = {
                    "ingredient": best_dal_label,
                    "confidence": max(best_dal_score, settings.min_confidence),
                }
            else:
                existing["confidence"] = max(float(existing.get("confidence", 0.0)), best_dal_score)

        corrected = list(by_name.values())
        corrected.sort(key=lambda item: item["confidence"], reverse=True)
        return corrected

    def detect_from_images(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        if not images:
            return []

        prompt_labels = self._prompt_labels
        text_features = self._text_features

        aggregate_scores: defaultdict[str, list[float]] = defaultdict(list)
        per_image_best: list[dict[str, Any]] = []
        red_likelihood = 0.0
        if images:
            red_likelihood = sum(self._estimate_red_food_likelihood(img) for img in images) / len(images)

        image_inputs = self.processor(images=[image.convert("RGB") for image in images], return_tensors="pt")
        image_inputs = {key: value.to(self.device) for key, value in image_inputs.items()}

        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            image_outputs = self.model.vision_model(
                pixel_values=image_inputs["pixel_values"],
                return_dict=True,
            )
            image_features = self.model.visual_projection(image_outputs.pooler_output)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            logits = (image_features @ text_features.T) * self.model.logit_scale.exp()
            probabilities = logits.softmax(dim=-1)

        for image_probabilities in probabilities:
            top_values, top_indices = torch.topk(image_probabilities, k=min(12, image_probabilities.shape[-1]))
            for score, index in zip(top_values.tolist(), top_indices.tolist(), strict=False):
                aggregate_scores[prompt_labels[index]].append(float(score))

        for image_probabilities in probabilities:
            best_score, best_index = torch.max(image_probabilities, dim=-1)
            per_image_best.append(
                {
                    "ingredient": prompt_labels[int(best_index.item())],
                    "confidence": float(best_score.item()),
                }
            )

        ranked = [
            {"ingredient": ingredient, "confidence": sum(scores) / len(scores)}
            for ingredient, scores in aggregate_scores.items()
            if (sum(scores) / len(scores)) >= settings.min_confidence
        ]

        # If thresholding removes valid single-image ingredients, backfill using each image's
        # strongest label so multi-image uploads still surface one ingredient per image.
        target_count = min(len(images), settings.max_detected_ingredients)
        if len(ranked) < target_count:
            existing = {item["ingredient"] for item in ranked}
            for candidate in sorted(per_image_best, key=lambda item: item["confidence"], reverse=True):
                ingredient = str(candidate.get("ingredient", "")).strip().lower()
                if not ingredient or ingredient in existing:
                    continue
                ranked.append(
                    {
                        "ingredient": ingredient,
                        "confidence": max(float(candidate.get("confidence", 0.0)), settings.min_confidence * 0.85),
                    }
                )
                existing.add(ingredient)
                if len(ranked) >= target_count:
                    break

        ranked = self._correct_common_misclassifications(ranked, aggregate_scores, red_likelihood)
        ranked.sort(key=lambda item: item["confidence"], reverse=True)
        return ranked[: settings.max_detected_ingredients]


@lru_cache(maxsize=1)
def get_detector() -> IngredientDetector:
    return IngredientDetector()
