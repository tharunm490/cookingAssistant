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
    "raw {ingredient} on a kitchen counter",
    "{ingredient} in a home kitchen",
]


@lru_cache(maxsize=1)
def get_clip_components() -> tuple[CLIPModel, CLIPProcessor, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(settings.clip_model_name)
    processor = CLIPProcessor.from_pretrained(settings.clip_model_name)
    model.to(device)
    model.eval()
    return model, processor, device


class IngredientDetector:
    def __init__(self) -> None:
        self.model, self.processor, self.device = get_clip_components()
        self.ingredients = self._build_ingredient_list()

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
        return prompts, labels

    def detect_from_images(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        if not images:
            return []

        prompts, prompt_labels = self._build_prompts()
        text_inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}

        with torch.no_grad():
            # Use the CLIP projection head explicitly so text/image features always share dims.
            text_outputs = self.model.text_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
                return_dict=True,
            )
            text_features = self.model.text_projection(text_outputs.pooler_output)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

        aggregate_scores: defaultdict[str, list[float]] = defaultdict(list)

        for image in images:
            image_inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
            image_inputs = {key: value.to(self.device) for key, value in image_inputs.items()}

            with torch.no_grad():
                image_outputs = self.model.vision_model(
                    pixel_values=image_inputs["pixel_values"],
                    return_dict=True,
                )
                image_features = self.model.visual_projection(image_outputs.pooler_output)
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                logits = (image_features @ text_features.T).squeeze(0) * self.model.logit_scale.exp()
                probabilities = logits.softmax(dim=-1)

            top_values, top_indices = torch.topk(probabilities, k=min(5, probabilities.shape[-1]))
            for score, index in zip(top_values.tolist(), top_indices.tolist(), strict=False):
                aggregate_scores[prompt_labels[index]].append(float(score))

        ranked = [
            {"ingredient": ingredient, "confidence": sum(scores) / len(scores)}
            for ingredient, scores in aggregate_scores.items()
            if (sum(scores) / len(scores)) >= settings.min_confidence
        ]
        ranked.sort(key=lambda item: item["confidence"], reverse=True)
        return ranked[: settings.max_detected_ingredients]


@lru_cache(maxsize=1)
def get_detector() -> IngredientDetector:
    return IngredientDetector()
