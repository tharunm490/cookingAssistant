const API_BASE = "http://127.0.0.1:8000";

const state = {
  files: [],
  detectedIngredients: [],
  recipe: null,
  audioUrl: null,
};

const dom = {
  fileInput: document.querySelector("#file-input"),
  dropzone: document.querySelector("#dropzone"),
  previewGrid: document.querySelector("#preview-grid"),
  detectButton: document.querySelector("#detect-button"),
  generateButton: document.querySelector("#generate-button"),
  voiceButton: document.querySelector("#voice-button"),
  ingredientList: document.querySelector("#ingredient-list"),
  recipeOutput: document.querySelector("#recipe-output"),
  nutritionOutput: document.querySelector("#nutrition-output"),
  audioOutput: document.querySelector("#audio-output"),
  status: document.querySelector("#status"),
  mealType: document.querySelector("#meal-type"),
  dietType: document.querySelector("#diet-type"),
  spiceLevel: document.querySelector("#spice-level"),
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
  if (!state.files.length) {
    dom.previewGrid.innerHTML = '<p class="status">No images selected yet. Drop pantry photos to begin.</p>';
    return;
  }

  state.files.forEach((file) => {
    const reader = new FileReader();
    reader.onload = () => {
      const tile = document.createElement("article");
      tile.className = "preview-tile";
      tile.innerHTML = `
        <img src="${reader.result}" alt="Preview of ${file.name}" />
        <span>${file.name}</span>
      `;
      dom.previewGrid.appendChild(tile);
    };
    reader.readAsDataURL(file);
  });
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

function renderRecipe() {
  if (!state.recipe) {
    dom.recipeOutput.innerHTML = '<p class="status">Generated recipe will appear here.</p>';
    dom.nutritionOutput.innerHTML = "";
    return;
  }

  const recipe = state.recipe;
  dom.recipeOutput.innerHTML = `
    <div class="recipe-meta">
      <span class="pill active">${recipe.cooking_time || "--"}</span>
      <span class="pill active">${recipe.servings || "--"} servings</span>
      <span class="pill active">${recipe.recipe_name}</span>
    </div>
    <h3 class="header-title" style="font-size:2rem;">${recipe.recipe_name}</h3>
    <p class="header-subtitle">A structured recipe assembled from detected ingredients and cooking preferences.</p>
    <h4>Ingredients with measurements</h4>
    <ul class="recipe-list">
      ${(recipe.ingredients_with_measurements || [])
        .map((item) => `<li><strong>${item.measurement}</strong> ${item.ingredient}</li>`)
        .join("")}
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

async function detectIngredients() {
  if (!state.files.length) {
    setStatus("Add at least one image first.");
    return;
  }

  const payload = new FormData();
  state.files.forEach((file) => payload.append("images", file));

  setStatus("Detecting ingredients with CLIP...");
  const response = await fetch(`${API_BASE}/detect-ingredients`, {
    method: "POST",
    body: payload,
  });

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

renderPreviews();
renderIngredients();
renderRecipe();
renderAudio();
