const API_BASE = "http://127.0.0.1:8000";

const state = {
  files: [],
  detectedIngredients: [],
  recipe: null,
  audioUrl: null,
  healthGoals: [],
};

const HEALTH_GOALS_OPTIONS = [
  { id: "weight_loss", label: "Weight Loss", emoji: "⚖️" },
  { id: "high_protein", label: "High Protein", emoji: "💪" },
  { id: "diabetic_friendly", label: "Diabetic Friendly", emoji: "🍎" },
  { id: "low_fat", label: "Low Fat", emoji: "💚" },
  { id: "heart_healthy", label: "Heart Healthy", emoji: "❤️" },
];

const dom = {
  fileInput: document.querySelector("#file-input"),
  dropzone: document.querySelector("#dropzone"),
  previewGrid: document.querySelector("#preview-grid"),
  clearImagesButton: document.querySelector("#clear-images-button"),
  detectButton: document.querySelector("#detect-button"),
  generateButton: document.querySelector("#generate-button"),
  voiceButton: document.querySelector("#voice-button"),
  newRecipeButton: document.querySelector("#new-recipe-button"),
  ingredientList: document.querySelector("#ingredient-list"),
  recipeOutput: document.querySelector("#recipe-output"),
  nutritionOutput: document.querySelector("#nutrition-output"),
  audioOutput: document.querySelector("#audio-output"),
  healthOptimizationOutput: document.querySelector("#health-optimization-output"),
  status: document.querySelector("#status"),
  mealType: document.querySelector("#meal-type"),
  dietType: document.querySelector("#diet-type"),
  spiceLevel: document.querySelector("#spice-level"),
  healthGoalsContainer: document.querySelector("#health-goals-container"),
};

function setStatus(message) {
  dom.status.textContent = message;
}

function promptLoginOrSignup(actionName) {
  setStatus(`Please login or signup first to ${actionName}.`);
  const goToLogin = window.confirm(`Please login or signup first to ${actionName}.\n\nPress OK for Login or Cancel for Signup.`);
  window.location.href = goToLogin ? "login.html" : "signup.html";
}

function getAuthToken() {
  return localStorage.getItem("sca_token");
}

function renderPreviews() {
  dom.previewGrid.innerHTML = "";
  dom.clearImagesButton.style.display = state.files.length > 0 ? "inline-block" : "none";
  
  if (!state.files.length) {
    dom.previewGrid.innerHTML = '<p class="status">No images selected yet. Drop pantry photos to begin.</p>';
    return;
  }

  state.files.forEach((file, index) => {
    const reader = new FileReader();
    reader.onload = () => {
      const tile = document.createElement("article");
      tile.className = "preview-tile";
      tile.innerHTML = `
        <div class="preview-tile-wrapper">
          <img src="${reader.result}" alt="Preview of ${file.name}" />
          <button class="delete-image-btn" data-index="${index}" title="Delete this image">✕</button>
        </div>
        <span class="preview-tile-name">${file.name}</span>
      `;
      
      const deleteBtn = tile.querySelector(".delete-image-btn");
      deleteBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        deleteImageByIndex(index);
      });
      
      dom.previewGrid.appendChild(tile);
    };
    reader.readAsDataURL(file);
  });
}

function deleteImageByIndex(index) {
  state.files = state.files.filter((_, i) => i !== index);
  state.detectedIngredients = [];
  state.recipe = null;
  state.audioUrl = null;
  renderPreviews();
  renderIngredients();
  renderRecipe();
  renderAudio();
  setStatus(`Image removed. ${state.files.length} image(s) remaining.`);
}

function clearAllImages() {
  state.files = [];
  state.detectedIngredients = [];
  state.recipe = null;
  state.audioUrl = null;
  renderPreviews();
  renderIngredients();
  renderRecipe();
  renderAudio();
  setStatus("All images cleared. Ready for new uploads.");
}

function renderIngredients() {
  dom.ingredientList.innerHTML = "";
  if (!state.detectedIngredients.length) {
    dom.ingredientList.innerHTML = '<span class="status">Detected ingredients will appear here.</span>';
    return;
  }

  state.detectedIngredients.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "chip good";
    chip.textContent = `${item.ingredient} ${(item.confidence * 100).toFixed(0)}%`;
    dom.ingredientList.appendChild(chip);
  });
}

function renderHealthGoals() {
  dom.healthGoalsContainer.innerHTML = "";
  
  HEALTH_GOALS_OPTIONS.forEach((goal) => {
    const label = document.createElement("label");
    label.className = "health-goal-checkbox";
    
    const isSelected = state.healthGoals.includes(goal.id);
    label.classList.toggle("selected", isSelected);
    
    label.innerHTML = `
      <input type="checkbox" value="${goal.id}" ${isSelected ? "checked" : ""} />
      <span class="checkbox-custom"></span>
      <span class="goal-label">${goal.emoji} ${goal.label}</span>
    `;
    
    const input = label.querySelector("input");
    input.addEventListener("change", () => {
      if (input.checked) {
        if (!state.healthGoals.includes(goal.id)) {
          state.healthGoals.push(goal.id);
        }
      } else {
        state.healthGoals = state.healthGoals.filter(g => g !== goal.id);
      }
      label.classList.toggle("selected", input.checked);
    });
    
    dom.healthGoalsContainer.appendChild(label);
  });
}

function renderRecipe() {
  if (!state.recipe) {
    dom.recipeOutput.innerHTML = '<p class="status">Generated recipe will appear here.</p>';
    dom.nutritionOutput.innerHTML = "";
    dom.healthOptimizationOutput.style.display = "none";
    return;
  }

  const recipe = state.recipe;
  const normalizeIngredient = (value) => String(value || "").trim().toLowerCase();
  const matchedIngredients = (recipe.matched_ingredients || [])
    .map((item) => normalizeIngredient(item))
    .filter(Boolean);
  const missingSet = new Set((recipe.missing_ingredients || []).map((item) => normalizeIngredient(item)));
  const measuredMissing = (recipe.ingredients_with_measurements || [])
    .map((item) => ({
      ingredient: normalizeIngredient(item.ingredient),
      measurement: String(item.measurement || "").trim(),
    }))
    .filter((item) => missingSet.has(item.ingredient));

  const matchedHtml = matchedIngredients.map((item) => `<li>✔ ${item}</li>`).join("");
  const extraMeasuredHtml = measuredMissing
    .map((item) => `<li>${item.ingredient} - ${item.measurement || "as needed"}</li>`)
    .join("");
  const extraFallbackHtml = (recipe.extra_ingredients || []).map((item) => `<li>${item}</li>`).join("");
  const source = recipe.generation_source || recipe.source || "unknown";
  const sourceLabel = source.startsWith("dataset")
    ? "Dataset Recipe"
    : source === "model"
      ? "AI Generated"
      : source === "fallback"
        ? "Safe Fallback"
        : source;

  dom.recipeOutput.innerHTML = `
    <div class="recipe-meta">
      <span class="pill active">${recipe.cooking_time || "--"}</span>
      <span class="pill active">${recipe.servings || "--"} servings</span>
      <span class="pill active">${recipe.recipe_name}</span>
    </div>
    <h3 class="header-title" style="font-size:2rem;">${recipe.recipe_name}</h3>
    <p class="header-subtitle">A structured recipe assembled from detected ingredients and cooking preferences.</p>
    <p class="header-subtitle"><strong>Source:</strong> ${sourceLabel}</p>
    <h4>Ingredients with measurements</h4>
    <ul class="recipe-list">
      ${(recipe.ingredients_with_measurements || [])
        .map((item) => `<li><strong>${item.measurement}</strong> ${item.ingredient}</li>`)
        .join("")}
    </ul>
    <h4>Matched Ingredients</h4>
    <ul class="recipe-list">
      ${matchedHtml || "<li>None</li>"}
    </ul>
    <h4>Extra Ingredients Needed</h4>
    <ul class="recipe-list">
      ${extraMeasuredHtml || extraFallbackHtml || "<li>None - You already have everything needed.</li>"}
    </ul>
    <h4>Steps</h4>
    <ol class="steps-list">
      ${(recipe.steps || []).map((step) => `<li>${step}</li>`).join("")}
    </ol>
  `;

  const nutrition = recipe.nutrition || {};
  dom.nutritionOutput.innerHTML = `
    <div class="stats-card"><strong>${nutrition.calories || "--"}</strong><span>calories</span></div>
    <div class="stats-card"><strong>${nutrition.protein || "--"}</strong><span>protein</span></div>
    <div class="stats-card"><strong>${nutrition.carbs || "--"}</strong><span>carbs</span></div>
    <div class="stats-card"><strong>${nutrition.fat || "--"}</strong><span>fat</span></div>
  `;
  
  // Show health optimization if health goals were selected
  if (state.healthGoals.length > 0) {
    const goalLabels = state.healthGoals.map(goal => {
      const goalObj = HEALTH_GOALS_OPTIONS.find(g => g.id === goal);
      return goalObj ? `${goalObj.emoji} ${goalObj.label}` : goal;
    }).join(", ");
    
    dom.healthOptimizationOutput.innerHTML = `
      <div class="health-optimization-badge">
        <h4 style="margin-top:0;">✅ Health Optimized</h4>
        <p><strong>Optimized for:</strong> ${goalLabels}</p>
        <p class="health-optimization-note">This recipe has been adjusted to align with your selected health goals, including ingredient selection, cooking methods, and nutrition profile.</p>
      </div>
    `;
    dom.healthOptimizationOutput.style.display = "block";
  } else {
    dom.healthOptimizationOutput.style.display = "none";
  }
}

function renderAudio() {
  if (!state.audioUrl) {
    dom.audioOutput.innerHTML = '<p class="status">Create voice narration after generating the recipe.</p>';
    return;
  }

  dom.audioOutput.innerHTML = `
    <audio controls src="${state.audioUrl}"></audio>
    <p class="status">Voice narration ready.</p>
  `;
}

async function downscaleImageFile(file, maxDimension = 1024, quality = 0.82) {
  if (!file || !file.type || !file.type.startsWith("image/")) {
    return file;
  }

  let imageBitmap = null;
  try {
    imageBitmap = await createImageBitmap(file);
    const originalWidth = imageBitmap.width;
    const originalHeight = imageBitmap.height;
    const longestSide = Math.max(originalWidth, originalHeight);
    if (longestSide <= maxDimension) {
      return file;
    }

    const scale = maxDimension / longestSide;
    const targetWidth = Math.max(1, Math.round(originalWidth * scale));
    const targetHeight = Math.max(1, Math.round(originalHeight * scale));
    const canvas = document.createElement("canvas");
    canvas.width = targetWidth;
    canvas.height = targetHeight;

    const context = canvas.getContext("2d", { alpha: false });
    if (!context) {
      return file;
    }
    context.drawImage(imageBitmap, 0, 0, targetWidth, targetHeight);

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
    if (!blob) {
      return file;
    }

    const fileName = file.name.replace(/\.[^.]+$/, "") || "image";
    return new File([blob], `${fileName}.jpg`, {
      type: "image/jpeg",
      lastModified: file.lastModified || Date.now(),
    });
  } catch {
    return file;
  } finally {
    if (imageBitmap && typeof imageBitmap.close === "function") {
      imageBitmap.close();
    }
  }
}

async function detectIngredients() {
  if (!state.files.length) {
    setStatus("Add at least one image first.");
    return;
  }

  const preparedFiles = [];
  for (const file of state.files) {
    preparedFiles.push(await downscaleImageFile(file));
  }

  const payload = new FormData();
  preparedFiles.forEach((file) => payload.append("images", file, file.name));

  setStatus("Detecting ingredients with CLIP...");
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 90000);
  let response;
  try {
    response = await fetch(`${API_BASE}/detect-ingredients`, {
      method: "POST",
      body: payload,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Detection is taking too long. Please try with fewer images.");
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    throw new Error("Ingredient detection failed.");
  }

  const data = await response.json();
  state.detectedIngredients = data.ingredients || [];
  renderIngredients();
  setStatus(`Detected ${state.detectedIngredients.length} ingredients.`);
}

async function generateRecipe() {
  const token = getAuthToken();
  if (!token) {
    promptLoginOrSignup("generate a recipe");
    return;
  }

  const ingredients = state.detectedIngredients.map((item) => item.ingredient);
  const response = await fetch(`${API_BASE}/generate-recipe`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      ingredients,
      meal_type: dom.mealType.value,
      diet: dom.dietType.value,
      spice_level: dom.spiceLevel.value,
      health_goals: state.healthGoals,
    }),
  });

  if (!response.ok) {
    if (response.status === 401) {
      localStorage.removeItem("sca_token");
      localStorage.removeItem("sca_user");
      promptLoginOrSignup("generate a recipe");
      return;
    }
    throw new Error("Recipe generation failed.");
  }

  state.recipe = await response.json();
  state.audioUrl = null;
  renderRecipe();
  renderAudio();
  setStatus("Recipe generated successfully.");
}

async function generateVoice() {
  if (!state.recipe) {
    setStatus("Generate a recipe first.");
    return;
  }

  const response = await fetch(`${API_BASE}/text-to-speech`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ recipe: state.recipe }),
  });

  if (!response.ok) {
    throw new Error("Audio generation failed.");
  }

  const data = await response.json();
  state.audioUrl = `${API_BASE}${data.audio_url}`;
  renderAudio();
  setStatus("Audio narration generated.");
}

function handleFiles(files) {
  state.files = Array.from(files || []);
  state.detectedIngredients = [];
  state.recipe = null;
  state.audioUrl = null;
  renderPreviews();
  renderIngredients();
  renderRecipe();
  renderAudio();
  setStatus(`${state.files.length} image(s) ready for analysis.`);
}

function resetAll() {
  // Reset state completely
  state.files = [];
  state.detectedIngredients = [];
  state.recipe = null;
  state.audioUrl = null;
  state.healthGoals = [];
  
  // Reset form inputs
  dom.mealType.value = "lunch";
  dom.dietType.value = "balanced";
  dom.spiceLevel.value = "medium";
  
  // Re-render everything
  renderPreviews();
  renderIngredients();
  renderRecipe();
  renderAudio();
  renderHealthGoals();
  
  setStatus("Ready for a new recipe! Upload images to begin.");
}

dom.fileInput.addEventListener("change", (event) => handleFiles(event.target.files));

dom.dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dom.dropzone.classList.add("dragover");
});

dom.dropzone.addEventListener("dragleave", () => {
  dom.dropzone.classList.remove("dragover");
});

dom.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dom.dropzone.classList.remove("dragover");
  handleFiles(event.dataTransfer.files);
});

dom.clearImagesButton.addEventListener("click", () => {
  clearAllImages();
});

dom.newRecipeButton.addEventListener("click", () => {
  resetAll();
});

dom.detectButton.addEventListener("click", async () => {
  try {
    await detectIngredients();
  } catch (error) {
    setStatus(error.message);
  }
});

dom.generateButton.addEventListener("click", async () => {
  try {
    if (!state.detectedIngredients.length) {
      await detectIngredients();
    }
    await generateRecipe();
  } catch (error) {
    setStatus(error.message);
  }
});

dom.voiceButton.addEventListener("click", async () => {
  try {
    await generateVoice();
  } catch (error) {
    setStatus(error.message);
  }
});

// Initialize
renderPreviews();
renderIngredients();
renderRecipe();
renderAudio();
renderHealthGoals();
