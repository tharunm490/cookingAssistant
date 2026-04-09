# Smart Cooking Assistant

Smart Cooking Assistant is a FastAPI + HTML/CSS/JS application that turns pantry photos into:

1. detected ingredients (CLIP image model),
2. a structured cooking recipe (Hugging Face inference with local fallback),
3. translated recipe output for selected UI languages (English/Hindi/Tamil/Telugu/Kannada),
3. optional voice narration (gTTS MP3),
4. saved recipe history for authenticated users.

---

## Features

- Multi-image ingredient detection with confidence scores.
- Recipe generation using ingredient list + user preferences.
- Automatic fallback recipe generation if Hugging Face is unavailable.
- Automatic translation of recipe response for supported languages.
- Text-to-speech output for generated recipes.
- User authentication with JWT.
- MySQL-backed storage for users, login history, recipes, images, and detected ingredients.
- Frontend pages for signup, login, dashboard, and profile-related screens.

---

## Project Layout

```text
SmartCookingAssistant/
   backend/
      main.py               # FastAPI routes
      clip_model.py         # CLIP ingredient detector
      recipe_generator.py   # Recipe generation + fallback logic
      tts_service.py        # gTTS narration
      db.py                 # MySQL connection helper
      config.py             # App settings and env loading
      requirements.txt

   frontend/
      index.html            # Main dashboard UI
      app.js                # Frontend API integration
      style.css
      login.html
      signup.html
      my-recipes.html
      profile.html
      settings.html
      about.html

   agile_ui_1.html         # Standalone styled dashboard variant
   generated_audio/        # Project-local audio folder (optional)
   uploaded_images/        # Project-local uploads folder (optional)
```

---

## Requirements

- Python 3.10+ recommended
- MySQL Server
- Internet access for Hugging Face inference (optional but recommended)
- Hugging Face token (optional; enables remote recipe model)

---

## Backend Dependencies

Dependencies are listed in `backend/requirements.txt`:

- fastapi
- uvicorn[standard]
- python-multipart
- pydantic
- transformers
- torch
- pillow
- huggingface_hub
- gtts
- aiofiles
- mysql-connector-python
- bcrypt
- python-jose
- deep-translator

---

## Quick Start (Windows PowerShell)

From the `SmartCookingAssistant` folder:

```powershell
py -3 -m venv agile_versiion1
.\agile_versiion1\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Run the API:

```powershell
uvicorn backend.main:app --reload --reload-dir backend --reload-exclude "agile_versiion1/*"
```

Open the UI in your browser:

- `frontend/index.html` (main frontend flow), or
- `agile_ui_1.html` (standalone dashboard-style page).

Frontend currently points to:

```text
http://127.0.0.1:8000
```

---

## Environment Variables

The backend loads values from OS environment and also from `SmartCookingAssistant/.env` (if present).

### Database

- `DB_HOST` (default: `localhost`)
- `DB_USER` (default: `root`)
- `DB_PASSWORD` (required)
- `DB_NAME` (default: `smart_cooking_assistant`)

### Authentication

- `JWT_SECRET_KEY` (default fallback exists, but set this in production)

### Model / AI

- `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` (optional but recommended)
- Recipe generation model: `Qwen/Qwen2.5-7B-Instruct` (configured in `backend/config.py`)
- Ingredient detection model: `openai/clip-vit-base-patch32`

#### Models Used In This Project

- `Qwen/Qwen2.5-7B-Instruct`
   - Used for recipe generation in `backend/recipe_generator.py` via Hugging Face Inference.
- `openai/clip-vit-base-patch32`
   - Used for ingredient detection in `backend/clip_model.py`.
- `gTTS` (Google Text-to-Speech engine)
   - Used for voice narration in `backend/tts_service.py`.

Note: If remote model inference fails or token/provider access is unavailable, recipe generation falls back to local rule-based logic.

### File Storage

- `SCA_AUDIO_DIR` (optional override for generated audio files)

Default storage paths if not overridden:

- Audio: `%LOCALAPPDATA%\\SmartCookingAssistant\\generated_audio`
- Uploaded images: `%LOCALAPPDATA%\\SmartCookingAssistant\\uploaded_images`

If `LOCALAPPDATA` is unavailable, backend falls back to project-adjacent directories.

---

## API Overview

Base URL: `http://127.0.0.1:8000`

### Public routes

- `GET /`
   - Health check.

- `POST /signup`
   - Creates user with hashed password.

- `POST /login`
   - Validates credentials.
   - Updates login history.
   - Returns JWT access token.

- `POST /detect-ingredients`
   - `multipart/form-data` with one or more `images` files.
   - Returns detected ingredient list and confidence.

- `POST /text-to-speech`
   - JSON body with recipe object.
   - Returns `audio_url` for generated narration.

### Authenticated routes (Bearer token required)

- `GET /me`
   - Returns authenticated user profile.

- `POST /generate-recipe`
   - JSON body with ingredients + preferences.
   - Requires Bearer token.
   - Returns structured recipe payload translated to selected request language.

- `POST /save-recipe`
   - Saves recipe and nutrition data.

- `POST /store-ingredients`
   - Stores normalized detected ingredients.

- `POST /upload-image`
   - Uploads and stores image path metadata.

- `GET /my-recipes`
   - Returns saved recipe list for current user.

- `PUT /update-password`
   - Updates user password after current password validation.

### Static mount routes

- `/audio/...` -> generated recipe narration files.
- `/uploads/...` -> uploaded images.

---

## Auth Flow (Frontend)

- Signup: `frontend/signup.html`
- Login: `frontend/login.html`

On successful login:

- token is stored in `localStorage` as `sca_token`
- user profile is stored as `sca_user`
- browser redirects to `../agile_ui_1.html`

---

## How Recipe Generation Works

1. Image detector predicts ingredients from uploaded images.
2. User preferences are collected (meal type, diet, spice level, etc.).
3. Recipe service builds a structured prompt and calls the configured Hugging Face model.
   Current configured recipe model: `Qwen/Qwen2.5-7B-Instruct`.
4. If model call fails or no token is configured, local fallback recipe logic is used.
5. Recipe response is translated for supported languages (`hi`, `ta`, `te`, `kn`) and left as-is for `en`.
6. Response is normalized into consistent JSON:
    - `recipe_name`
    - `input_ingredients`
    - `extra_ingredients`
    - `ingredients_with_measurements`
    - `steps`
    - `cooking_time`
    - `servings`
    - `nutrition`

---

## Database Notes

The backend expects MySQL tables for users, login history, recipes, detected ingredients, and uploaded images.

At minimum, ensure your schema supports these route operations:

- create/select user rows
- insert login history
- insert/select recipes by `user_id`
- insert detected ingredient snapshots
- insert uploaded image metadata

If these tables are missing, authenticated and persistence-related routes will fail.

---

## Troubleshooting

### `DB_PASSWORD is not set`

Set `DB_PASSWORD` in environment or in `.env` at project root.

### `Invalid or expired token`

Login again to refresh `sca_token` and send header:

```text
Authorization: Bearer <token>
```

### Recipe still works without HF token

This is expected. The app uses fallback recipe generation when remote inference is unavailable.

### Translation not appearing in non-English language

- Confirm request body includes `language` as one of: `hi`, `ta`, `te`, `kn`.
- Confirm `deep-translator` is installed in the same virtual environment used by `uvicorn`.
- Translation library requires internet access to fetch translated text.

### Audio or upload files are not in the repository folder

Expected default behavior. Files are written under `%LOCALAPPDATA%\SmartCookingAssistant\...` unless overridden.

---

## Development Notes

- CORS is currently permissive (`*`) for development.
- Replace default JWT secret and tighten CORS before production deployment.
- Keep large virtual environments out of version control.

---

## License

No license file is currently included in this repository.
Add a license if you plan to distribute or open-source this project.
