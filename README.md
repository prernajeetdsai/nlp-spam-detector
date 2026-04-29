# NLP Spam Detector with Explainability

An end-to-end spam email classification system built with **TF-IDF + Logistic Regression**, explained using **SHAP (SHapley Additive exPlanations)**, served via a **Flask REST API**, and fully containerised with **Docker**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [API Reference](#api-reference)
- [Explainability Methods](#explainability-methods)
- [Running Tests](#running-tests)
- [Docker](#docker)
- [File-by-File Explanation](#file-by-file-explanation)
- [Web UI](#web-ui)

---

## Project Overview

This project classifies emails or messages as **spam** or **ham (not spam)** and explains *why* the model made each decision. Instead of just returning a label, the API tells you exactly which words pushed the model toward that decision — making the model transparent and trustworthy.

**Tech Stack**
- **Model:** TF-IDF Vectorizer + Logistic Regression (scikit-learn)
- **Explainability:** SHAP LinearExplainer + Feature Importance
- **API:** Flask (REST)
- **Containerisation:** Docker
- **Testing:** pytest

---

## Project Structure

```
nlp-spam-detector/
│
├── app.py                          # Flask REST API — 3 endpoints
├── model.py                        # Train the model & load for inference
├── explain.py                      # SHAP & feature importance explainability
├── emails.csv                      # Dataset (5728 emails, spam/ham labels)
├── setup.py                        # Makes the project pip-installable
├── requirements.txt                # All Python dependencies
├── Dockerfile                      # Container build instructions
│
├── src/                            # Core source package
│   ├── __init__.py
│   ├── exception.py                # Custom exception with file + line info
│   ├── logger.py                   # Centralised logging to console + file
│   ├── utils.py                    # Shared helpers: clean_text, save/load object
│   │
│   ├── components/                 # Individual ML pipeline steps
│   │   ├── __init__.py
│   │   ├── data_ingestion.py       # Reads CSV, splits train/test, saves to artifacts/
│   │   └── model_trainer.py        # Trains TF-IDF + LR, evaluates, saves model
│   │
│   └── pipeline/                   # Orchestration layer
│       ├── __init__.py
│       ├── train_pipeline.py       # Runs ingestion → training end-to-end
│       └── predict_pipeline.py     # Loads model, exposes predict()
│
├── tests/                          # Unit & integration tests
│   ├── __init__.py
│   └── test_app.py                 # 13 tests covering utils, pipeline, API
│
├── artifacts/                      # Auto-created when you train
│   ├── model.pkl                   # Saved trained model
│   ├── vectorizer.pkl              # Saved TF-IDF vectorizer
│   ├── raw.csv                     # Copy of full dataset
│   ├── train.csv                   # Training split (80%)
│   └── test.csv                    # Test split (20%)
│
└── logs/                           # Auto-created at runtime
    └── YYYY_MM_DD_HH_MM_SS.log     # Timestamped log file
```

> **Note:** `artifacts/` and `logs/` are generated automatically when you run the code. They are not included in the repository.

---

## How It Works

```
emails.csv
    │
    ▼
data_ingestion.py        ← reads CSV, creates train/test split
    │
    ▼
model_trainer.py         ← cleans text → TF-IDF → Logistic Regression
    │
    ▼
artifacts/
  model.pkl
  vectorizer.pkl
    │
    ▼
app.py (Flask API)       ← loads model on startup
    │
    ├── POST /predict          ← classify + SHAP explanation
    └── POST /predict/explain  ← classify + choose explanation method
```

**Text preprocessing pipeline (inside `src/utils.py`):**
1. Lowercase everything
2. Remove punctuation and special characters
3. Collapse multiple spaces into one
4. Feed into TF-IDF vectorizer (10,000 features, unigrams + bigrams)

---

## Dataset

- **Source:** [Spam Email Dataset — Kaggle](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset/data)
- **File:** `emails.csv`
- **Size:** 5,728 emails
- **Columns:** `text` (email content), `spam` (1 = spam, 0 = ham)
- **Class distribution:**

| Label | Count | Percentage |
|-------|-------|------------|
| Ham (not spam) | 4,360 | 76.1% |
| Spam | 1,368 | 23.9% |

---

## Model Performance

Trained on 80% of data (4,582 emails), evaluated on 20% (1,146 emails):

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Ham   | 0.98      | 1.00   | 0.99     | 872     |
| Spam  | 0.98      | 0.94   | 0.96     | 274     |
| **Overall Accuracy** | | | **98.25%** | 1,146 |

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

Or install the project as a package (uses `setup.py`):

```bash
pip install -e .
```

---

## How to Run

### Step 1 — Train the model

```bash
python model.py
```

This will:
- Load and clean `emails.csv`
- Train the TF-IDF + Logistic Regression model
- Print accuracy and classification report
- Save `artifacts/model.pkl` and `artifacts/vectorizer.pkl`

Expected output:
```
INFO: Loading dataset from emails.csv
INFO: Fitting TF-IDF vectorizer …
INFO: Training Logistic Regression …
INFO: Accuracy: 0.9825
INFO: Saved object → artifacts/model.pkl
INFO: Saved object → artifacts/vectorizer.pkl
```

### Step 2 — Start the API

```bash
python app.py
```

Expected output:
```
INFO: Loading model artefacts …
INFO: Model ready.
* Running on http://0.0.0.0:5000
```

Leave this terminal running. Open a **second terminal** to test the API.

### Alternative — Run the full pipeline

Instead of `python model.py`, you can run the full structured pipeline:

```bash
# As a module (recommended)
python -m src.pipeline.train_pipeline

# As a script
python src/pipeline/train_pipeline.py
```

This runs data ingestion + training together and saves split CSVs to `artifacts/`.

---

## API Reference

### `GET /health`

Checks if the API is running.

**Request:**
```bash
curl.exe http://localhost:5001/health
```

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /predict`

Classifies text and returns a prediction with a SHAP explanation.

**Request body:**
```json
{ "text": "Your email or message here" }
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d "{\"text\": \"Congratulations! You won a free gift card. Click here now!\"}"
```

**Mac / Linux:**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won a free gift card. Click here now!"}'
```

**Response:**
```json
{
  "text": "Congratulations! You won a free gift card. Click here now!",
  "label": "spam",
  "confidence": 0.7844,
  "spam_probability": 0.7844,
  "explanation": {
    "method": "shap",
    "top_spam_words": [
      { "word": "click",     "shap_value": 0.7342 },
      { "word": "gift card", "shap_value": 0.3673 },
      { "word": "claim",     "shap_value": 0.3273 },
      { "word": "free",      "shap_value": 0.2196 }
    ],
    "top_ham_words": [
      { "word": "congratulations", "shap_value": -0.2741 }
    ],
    "base_value": -0.9583
  }
}
```

**Response fields explained:**

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | `"spam"` or `"ham"` |
| `confidence` | float | Probability of the predicted class (0–1) |
| `spam_probability` | float | Always the spam probability regardless of label |
| `explanation.top_spam_words` | list | Words that pushed the model toward spam |
| `explanation.top_ham_words` | list | Words that pushed the model toward ham |
| `explanation.base_value` | float | Model's baseline log-odds before seeing any words |

---

### `POST /predict/explain`

Same as `/predict` but lets you choose the explanation method and number of words returned.

**Request body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | ✅ Yes | — | The text to classify |
| `method` | string | ❌ No | `"shap"` | `"shap"` or `"feature_importance"` |
| `top_n` | int | ❌ No | `10` | Number of top words to return |

**Windows (PowerShell):**
```powershell
curl.exe -X POST http://localhost:5001/predict/explain -H "Content-Type: application/json" -d "{\"text\": \"Hey, are we still meeting tomorrow at 3pm?\", \"method\": \"feature_importance\", \"top_n\": 5}"
```

**Response:**
```json
{
  "text": "Hey, are we still meeting tomorrow at 3pm?",
  "label": "ham",
  "confidence": 0.8405,
  "spam_probability": 0.1595,
  "explanation": {
    "method": "feature_importance",
    "top_spam_words": [
      { "word": "hey", "score": 0.2952 }
    ],
    "top_ham_words": [
      { "word": "meeting",  "score": 0.6708 },
      { "word": "tomorrow", "score": 0.3278 }
    ]
  }
}
```

---

### Error Responses

| Situation | Status Code | Response |
|-----------|-------------|----------|
| Missing `text` field | `400` | `{"error": "Request body must contain a 'text' field."}` |
| Empty `text` field | `400` | `{"error": "'text' field must not be empty."}` |
| Invalid method | `400` | `{"error": "method must be 'shap' or 'feature_importance'."}` |

---

## Explainability Methods

### SHAP (default)

Uses `shap.LinearExplainer` to compute exact **Shapley values** for every word in the input.

- A **positive SHAP value** means the word pushed the model toward **spam**
- A **negative SHAP value** means the word pushed the model toward **ham**
- The `base_value` is the model's prediction before it sees any words (the average log-odds across all training data)

Example interpretation:
```
"click" has shap_value: 0.73
```
This means the word "click" added 0.73 to the log-odds of being spam — a strong spam signal.

### Feature Importance

A faster alternative. Computes `TF-IDF score × model coefficient` for each word in the input.

- Positive score → spam signal
- Higher absolute score → stronger influence

Use this when you want a quick explanation without the overhead of SHAP.

| Method | Speed | Accuracy of explanation |
|--------|-------|------------------------|
| `shap` | Slower (first call builds explainer) | Exact Shapley values |
| `feature_importance` | Fast | Good approximation |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Expected output:
```
tests/test_app.py::TestCleanText::test_lowercases_text          PASSED
tests/test_app.py::TestCleanText::test_removes_punctuation      PASSED
tests/test_app.py::TestCleanText::test_collapses_whitespace     PASSED
tests/test_app.py::TestCleanText::test_handles_empty_string     PASSED
tests/test_app.py::TestCleanText::test_handles_numbers          PASSED
tests/test_app.py::TestCleanText::test_strips_leading_trailing_spaces  PASSED
tests/test_app.py::TestPredictPipeline::test_predict_returns_spam      PASSED
tests/test_app.py::TestPredictPipeline::test_predict_returns_ham       PASSED
tests/test_app.py::TestPredictPipeline::test_predict_output_keys       PASSED
tests/test_app.py::TestFlaskAPI::test_health_endpoint           PASSED
tests/test_app.py::TestFlaskAPI::test_predict_missing_text      PASSED
tests/test_app.py::TestFlaskAPI::test_predict_empty_text        PASSED
tests/test_app.py::TestFlaskAPI::test_predict_valid_request     PASSED

13 passed in 6.67s
```

**What the tests cover:**

| Test Class | What it tests |
|------------|---------------|
| `TestCleanText` | Text preprocessing — lowercase, punctuation removal, whitespace, edge cases |
| `TestPredictPipeline` | Prediction returns correct label, probability, and output keys |
| `TestFlaskAPI` | All 3 endpoints — health check, missing/empty input (400), valid prediction (200) |

---

## Docker

Build the image and run the container with one command each:

```bash
# Build (also trains the model inside the container)
docker build -t nlp-app .

# Run on port 5001 (UI) — maps to the app's internal port
docker run -p 5001:5001 nlp-app
```

The Dockerfile:
1. Starts from `python:3.11-slim`
2. Installs all dependencies from `requirements.txt`
3. Copies all source files
4. Runs `python model.py` to train the model at build time
5. Starts the Flask API when the container runs

Once the container is running, test it exactly the same way as local:
```powershell
curl.exe http://localhost:5001/health
```

---

## File-by-File Explanation

### Root files

| File | Purpose |
|------|---------|
| `app.py` | Flask REST API. Loads the model once at startup and serves 3 endpoints: `/health`, `/predict`, `/predict/explain` |
| `model.py` | Entry point for training. Calls `src/` components, saves artefacts to `artifacts/`. Also exposes `load_model()` and `predict()` for use by the API |
| `explain.py` | Contains both SHAP and feature importance explanation logic. Imported directly by `app.py` |
| `emails.csv` | Raw dataset — 5,728 emails with `text` and `spam` columns |
| `setup.py` | Makes the project installable as a Python package via `pip install -e .` |
| `requirements.txt` | Pinned dependency versions to ensure reproducibility |
| `Dockerfile` | Instructions to build a self-contained Docker image |

### `src/` package

| File | Purpose |
|------|---------|
| `src/exception.py` | Custom `SpamDetectorException` class that captures the exact filename and line number where any error occurred — makes debugging much faster |
| `src/logger.py` | Sets up a single shared logger that writes to both the terminal and a timestamped `.log` file in `logs/`. All other files import from here instead of setting up their own logging |
| `src/utils.py` | Three shared functions used everywhere: `clean_text()` (preprocessing), `save_object()` (pickle to disk), `load_object()` (load from disk). Centralising these avoids duplicating code |
| `src/components/data_ingestion.py` | Responsible for one thing only: reading `emails.csv`, splitting it 80/20, and saving `raw.csv`, `train.csv`, `test.csv` to `artifacts/` |
| `src/components/model_trainer.py` | Responsible for one thing only: taking train/test CSVs, fitting TF-IDF + Logistic Regression, evaluating accuracy, and saving `model.pkl` + `vectorizer.pkl` |
| `src/pipeline/train_pipeline.py` | Glues `DataIngestion` and `ModelTrainer` together. Run this to do the full training flow in one command |
| `src/pipeline/predict_pipeline.py` | Loads saved artefacts and exposes a `predict()` method. Used by `app.py` to serve real-time predictions |

### `tests/`

| File | Purpose |
|------|---------|
| `tests/test_app.py` | 13 unit and integration tests. Uses `unittest.mock` to fake the model so tests run without needing `model.pkl` on disk |

### Auto-generated (not in repository)

| Folder/File | Created by | Contents |
|-------------|------------|---------|
| `artifacts/model.pkl` | `python model.py` | Trained Logistic Regression model |
| `artifacts/vectorizer.pkl` | `python model.py` | Fitted TF-IDF vectorizer |
| `artifacts/train.csv` | `train_pipeline.py` | 80% training split |
| `artifacts/test.csv` | `train_pipeline.py` | 20% test split |
| `logs/*.log` | Any file that imports `src/logger.py` | Timestamped log of all INFO messages |

---

## Web UI

The project includes a browser-based interface for interacting with the spam detector — no curl commands needed.

### Accessing the UI

Once the Flask app (or Docker container) is running, open your browser and navigate to:

```
http://localhost:5001/
```

From the UI you can:
- Paste or type any email/message text
- Click **Predict** to instantly classify it as spam or ham
- See the confidence score and SHAP-based word explanations rendered visually

> **Note:** The UI runs on **port 5001**. Make sure to use `5001` in your browser, not `5000` (which is the raw API port).

### Running with Docker on port 5001

```bash
# Build the image
docker build -t nlp-app .

# Run and map container port → host port 5001
docker run -p 5001:5001 nlp-app
```

Then visit `http://localhost:5001/` in your browser.

### Project Structure (with templates)

```
nlp-spam-detector/
│
├── app.py                          # Flask REST API + serves the UI
├── model.py                        # Train the model & load for inference
├── explain.py                      # SHAP & feature importance explainability
├── emails.csv                      # Dataset (5728 emails, spam/ham labels)
├── setup.py                        # Makes the project pip-installable
├── requirements.txt                # All Python dependencies
├── Dockerfile                      # Container build instructions
│
├── templates/                      # HTML templates for the web UI
│   └── index.html                  # Main UI page served at http://localhost:5001/
│
├── src/                            # Core source package
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   ├── utils.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   └── model_trainer.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py
│       └── predict_pipeline.py
│
├── tests/
│   ├── __init__.py
│   └── test_app.py
│
├── artifacts/                      # Auto-created when you train
│   ├── model.pkl
│   ├── vectorizer.pkl
│   ├── raw.csv
│   ├── train.csv
│   └── test.csv
│
└── logs/
    └── YYYY_MM_DD_HH_MM_SS.log
```

