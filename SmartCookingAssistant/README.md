# Smart Cooking Assistant

Smart Cooking Assistant is a FastAPI + HTML/CSS/JS app that turns pantry photos into recipes.

## What It Does

- Detects ingredients from images (CLIP)
- Matches recipes from `Cleaned_Indian_Food_Dataset.csv` using a 9-phase retrieval pipeline
- Falls back to AI recipe generation if dataset match is weak
- Supports translation: English, Hindi, Tamil, Telugu, Kannada
- Generates recipe audio narration (gTTS)
- Supports user accounts and saved recipes (JWT + MySQL)

## Tech Stack

- Backend: FastAPI, Python
- Frontend: HTML/CSS/JS
- Database: MySQL
- Models: CLIP (`openai/clip-vit-base-patch32`), Qwen (`Qwen/Qwen2.5-7B-Instruct` fallback)

## Project Structure

```text
SmartCookingAssistant/
  backend/
    main.py
    recipe_generator.py
    clip_model.py
    translation_service.py
    tts_service.py
    db.py
    config.py
    requirements.txt
  frontend/
    index.html
    app.js
    style.css
    login.html
    signup.html
    my-recipes.html
    profile.html
    settings.html
    about.html
  agile_ui_1.html
  Cleaned_Indian_Food_Dataset.csv
```

## Requirements

- Python 3.10+
- MySQL 8+
- Optional: Hugging Face token for AI fallback

## Quick Start (Windows PowerShell)

```powershell
cd SmartCookingAssistant
py -3 -m venv agile_versiion1
.\agile_versiion1\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Create `.env` in `SmartCookingAssistant`:

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password_here
DB_NAME=smart_cooking_assistant
JWT_SECRET_KEY=your_secret_key_here
HF_TOKEN=your_huggingface_token_here
```

Run backend:

```powershell
.\agile_versiion1\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open frontend:

- `frontend/index.html` (main UI)
- `agile_ui_1.html` (alternate UI)

## Core API Endpoints

Base URL: `http://127.0.0.1:8000`

- `GET /` health check
- `POST /signup`
- `POST /login`
- `POST /detect-ingredients`
- `POST /generate-recipe` (auth required)
- `POST /save-recipe` (auth required)
- `GET /my-recipes` (auth required)
- `POST /text-to-speech`

## Recipe Retrieval (Short)

The dataset matcher uses a 9-phase flow:

1. Parse user intent
2. Clean titles
3. Filter by dish type
4. Penalize composite dishes
5. Score candidates
6. Match ingredients
7. Apply threshold
8. AI fallback
9. Safe template fallback

## Important Notes

- `DB_PASSWORD` must be set.
- If `HF_TOKEN` is missing, AI fallback may be limited.
- Uploaded images and generated audio can be stored under local app data paths by backend config.

## Additional Documentation

- `PHASE_RETRIEVAL_SYSTEM.md` (full pipeline details)
- `TESTING_GUIDE.md` (how to test)
- `TEST_REPORT.md` (latest test results)
- `BEFORE_AFTER_ANALYSIS.md` (improvement summary)

## Version

- Version: 2.0
- Last Updated: April 11, 2026
